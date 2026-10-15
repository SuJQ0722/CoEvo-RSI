"""Evolvable generative-LLM call component for the RBAC cascade."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from src.training.message_schema import (
    canonicalize_assistant_message,
    tool_result_message,
)

from .policy_guard import (
    grant_grounding_context,
    grant_is_grounded,
    inaccessible_tables,
    named_unavailable_column,
)
from .prompts import MAX_TOOL_RESULT_CHARS, MODEL_PROMPT_CHAR_BUDGET
from .tools.final_answer import (
    ACCESS_DENIED,
    ACCESS_GRANTED,
    SCHEMA as FINAL_ANSWER_SCHEMA,
    execute as execute_final_answer,
)
from .tools.test_sql import SCHEMA as TEST_SQL_SCHEMA, execute as execute_test_sql


@dataclass(frozen=True)
class LLMCallResult:
    response: str
    messages: list[dict[str, Any]]
    usage: dict[str, int]
    latency_seconds: float
    sql: str | None = None
    turns: int = 0
    error: str = ""


class LLMCaller:
    """Multi-turn agent loop with test_sql and final_answer tools."""

    tools = [FINAL_ANSWER_SCHEMA, TEST_SQL_SCHEMA]

    _FINAL_REMINDER = (
        "Protocol checkpoint: respond now with exactly one final_answer tool call. "
        "Use access_granted=false if any column required by the semantically correct "
        "query is unauthorized; otherwise include the best verified SQL and use true. "
        "Do not call test_sql again."
    )

    _TOOL_REMINDER = (
        "Protocol correction: your preceding response did not call a tool. Reassess "
        "briefly, then call final_answer, or call test_sql only if one execution check "
        "is still essential. Do not answer with prose alone."
    )

    # On the llm_fallback path an exhausted loop is not converted into an abort or
    # a silent default.  The most recent test_sql submission that cleared both the
    # whole-table veto and the grant-side citation check is returned as a formal
    # grant with its SQL; with no such submission the decision fails closed to a
    # formal denial.  Both terminal branches report error="".
    #
    # A denial on the llm_fallback path must cite header-grounded evidence; one
    # bounded reconsideration turn replaces an unsupported refusal. It is used at
    # most once per call, so the loop stays bounded and a second refusal is
    # returned exactly as the model decided it.
    _DENIAL_REMINDER = (
        "Denial evidence required: access_granted=false is accepted only when you also "
        "pass unauthorized_column naming one schema column that the role's Accessible "
        "Columns policy does not grant in any table that declares it. If every column "
        "your correct query needs is accessible, call final_answer(access_granted=true, "
        "sql=<your SQL>) now. Otherwise call final_answer(access_granted=false, "
        "unauthorized_column=<the blocked column>). Call final_answer exactly once more."
    )

    # A normal llm_fallback grant whose SQL cites a table or column the structured
    # header withholds is not accepted straight away.  The model gets exactly one
    # bounded chance to repair it, with the evidence and grant-preserving guidance
    # riding inside the rejected tool observation; a second ungrounded grant fails
    # closed to a formal denial.  This is the over-refusal countermeasure: a
    # definite ``None`` verdict abstains and is returned as the model decided, and
    # the re-ask keeps the grant option open instead of forcing a refusal.  The
    # guidance names the header's declared tables and the exact offending citation,
    # and, when the same withheld column is granted elsewhere, the declared
    # alternative table, so a shortcut-table plan can be rewritten instead of
    # resubmitted verbatim.
    _GRANT_GROUNDING_REMINDER = (
        "Grant not accepted: the submitted SQL reads {evidence}, which the role's "
        "Accessible Columns policy withholds. Declared tables in this schema: "
        "{declared_tables}. {alternatives}Either (a) call final_answer again with "
        "access_granted=true and sql rewritten so that every table and column it "
        "reads is authorized, or (b) if the semantically correct query genuinely "
        "needs that withheld column and no declared table grants it, call "
        "final_answer(access_granted=false, unauthorized_column=<that column>). "
        "Do not resubmit the same SQL and do not call test_sql again."
    )

    _GRANT_GROUNDING_ALTERNATIVES = (
        "The withheld column {column} is granted in declared table(s) {tables}; "
        "read the value from one of those tables only if it genuinely stores the "
        "value the query needs. "
    )

    # Shrink floor for an individual tool result or assistant prose block; only
    # oversized transcripts are touched and the original request/policy text is
    # never a shrink candidate.
    _MIN_SHRUNK_CHARS = 400
    _SHRINK_ORDER = ("tool", "assistant")

    def __init__(self, max_turns: int = 10, max_sql_tests: int = 8):
        self.max_turns = max_turns
        self.max_sql_tests = max_sql_tests

    @classmethod
    def _message_chars(cls, message: dict[str, Any]) -> int:
        content = message.get("content")
        size = len(content) if isinstance(content, str) else 0
        tool_calls = message.get("tool_calls")
        if tool_calls:
            try:
                size += len(json.dumps(tool_calls, default=str))
            except (TypeError, ValueError):
                size += len(str(tool_calls))
        return size

    @classmethod
    def _transcript_chars(cls, transcript: list[dict[str, Any]]) -> int:
        return sum(cls._message_chars(message) for message in transcript)

    @staticmethod
    def _bound_tool_result(payload: str) -> str:
        """Keep a single tool result under MAX_TOOL_RESULT_CHARS, staying valid JSON."""
        if not isinstance(payload, str) or len(payload) <= MAX_TOOL_RESULT_CHARS:
            return payload
        preview = payload[: max(0, MAX_TOOL_RESULT_CHARS - 200)]
        return json.dumps(
            {
                "truncated": True,
                "note": (
                    "Result shortened to protect the prompt budget; check joins and "
                    "identifiers against the schema instead of fetching more rows."
                ),
                "preview": preview,
            }
        )

    @classmethod
    def _bounded_transcript(
        cls, transcript: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Shrink tool-result text, then assistant prose, oldest-first.

        Message order, system prompts, the request text and the tool-call/tool-result
        pairing are preserved; only ``tool`` and ``assistant`` content is rewritten,
        and only when the assembled transcript exceeds the documented budget.
        """
        if cls._transcript_chars(transcript) <= MODEL_PROMPT_CHAR_BUDGET:
            return transcript
        bounded = [dict(message) for message in transcript]
        while cls._transcript_chars(bounded) > MODEL_PROMPT_CHAR_BUDGET:
            shrunk = False
            for role in cls._SHRINK_ORDER:
                for message in bounded:
                    if cls._transcript_chars(bounded) <= MODEL_PROMPT_CHAR_BUDGET:
                        break
                    if message.get("role") != role:
                        continue
                    content = message.get("content")
                    if not isinstance(content, str) or len(content) <= cls._MIN_SHRUNK_CHARS:
                        continue
                    keep = max(cls._MIN_SHRUNK_CHARS, len(content) // 2)
                    message["content"] = content[:keep]
                    shrunk = True
                if cls._transcript_chars(bounded) <= MODEL_PROMPT_CHAR_BUDGET:
                    break
            if not shrunk:
                break
        return bounded

    @staticmethod
    def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
        value = (tool_call.get("function") or {}).get("arguments", {})
        if isinstance(value, dict):
            return value
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    _MAX_DECLARED_TABLES_IN_GUIDANCE = 30

    @classmethod
    def _grant_grounding_guidance(cls, context: dict[str, Any]) -> str:
        """Compose header-grounded repair text for a definite ungrounded grant.

        The context is produced by ``policy_guard.grant_grounding_context`` and is
        non-empty only for a definite ``False`` verdict, so this text can never be
        assembled from an abstention or an authorized grant.  The declared-table
        list is capped so a pathological schema cannot crowd the tool observation.
        """
        declared = [str(name) for name in context.get("declared_tables") or []]
        if len(declared) > cls._MAX_DECLARED_TABLES_IN_GUIDANCE:
            declared = declared[: cls._MAX_DECLARED_TABLES_IN_GUIDANCE] + ["..."]
        alternatives = [
            str(name) for name in context.get("alternative_tables") or []
        ]
        alternative_text = ""
        if alternatives:
            alternative_text = cls._GRANT_GROUNDING_ALTERNATIVES.format(
                column=str(context.get("column") or context.get("evidence") or ""),
                tables=", ".join(alternatives),
            )
        return cls._GRANT_GROUNDING_REMINDER.format(
            evidence=str(context.get("evidence") or ""),
            declared_tables=", ".join(declared) if declared else "(none declared)",
            alternatives=alternative_text,
        )

    def call(
        self,
        model: Any,
        messages: list[dict[str, str]],
        *,
        db_path: str | None = None,
        policy_instruction: str | None = None,
        require_denial_evidence: bool = False,
        recover_exhausted_grant: bool = False,
        verify_grant_grounding: bool = False,
    ) -> LLMCallResult:
        started = time.perf_counter()
        transcript = list(messages)
        total_usage: dict[str, int] = {}
        sql_result: str | None = None
        turns = 0
        sql_tests = 0
        denial_reconsider_used = False
        grant_grounding_reask_used = False
        recovered_sql: str | None = None
        try:
            for turn in range(1, self.max_turns + 1):
                turns = turn
                assistant = dict(
                    model.complete(self._bounded_transcript(transcript), self.tools)
                )
                usage = {
                    key: int(value or 0)
                    for key, value in dict(
                        assistant.pop("_coevox_usage", {}) or {}
                    ).items()
                }
                for key, value in usage.items():
                    total_usage[key] = total_usage.get(key, 0) + value
                assistant = canonicalize_assistant_message(assistant)
                tool_calls = assistant.get("tool_calls") or []
                transcript.append(assistant)

                if not tool_calls:
                    # A missing tool call accounted for half of the operational
                    # failures in the baseline. Keep the failure bounded by the
                    # ordinary turn limit, but give the model a chance to repair the
                    # protocol instead of converting it immediately into a denial.
                    transcript.append({"role": "user", "content": self._TOOL_REMINDER})
                    continue

                needs_final_reminder = False
                for tool_call in tool_calls:
                    function = tool_call.get("function") or {}
                    name = function.get("name", "")
                    args = self._arguments(tool_call)
                    call_id = tool_call.get("id") or f"rbac_{name}_{turn}"
                    tool_call["id"] = call_id

                    if name == "final_answer":
                        requested_grant = args.get("access_granted")
                        if isinstance(requested_grant, str):
                            requested_grant = requested_grant.strip().lower() == "true"
                        sql_argument = args.get("sql")
                        blocked_tables = (
                            inaccessible_tables(policy_instruction, sql_argument)
                            if requested_grant is True
                            and isinstance(policy_instruction, str)
                            and isinstance(sql_argument, str)
                            else ()
                        )
                        if blocked_tables:
                            # A table omitted from Accessible Columns contributes no
                            # authorized columns. Never return executable SQL that
                            # reads it. This narrow check does not guess which columns
                            # a natural-language request might require.
                            tool_response, response, _sql = execute_final_answer(
                                {"access_granted": False}
                            )
                            transcript.append(
                                tool_result_message(
                                    tool_call, self._bound_tool_result(tool_response)
                                )
                            )
                            return LLMCallResult(
                                response=response or ACCESS_DENIED,
                                messages=transcript,
                                usage=total_usage,
                                latency_seconds=time.perf_counter() - started,
                                sql=None,
                                turns=turns,
                            )
                        if requested_grant is True and verify_grant_grounding:
                            grounding = grant_is_grounded(
                                policy_instruction, sql_argument
                            )
                            if grounding is False:
                                # Tri-state check: only a definite False is acted
                                # on.  None (unparseable header/SQL) abstains and
                                # falls through to the model's own grant.
                                context = grant_grounding_context(
                                    policy_instruction, sql_argument
                                )
                                evidence = str(context.get("evidence") or "")
                                if not grant_grounding_reask_used:
                                    grant_grounding_reask_used = True
                                    transcript.append(
                                        tool_result_message(
                                            tool_call,
                                            self._bound_tool_result(
                                                json.dumps(
                                                    {
                                                        "status": "not_accepted",
                                                        "error": (
                                                            "the grant cites table "
                                                            "or column evidence the "
                                                            "role's Accessible "
                                                            "Columns policy withholds"
                                                        ),
                                                        "evidence": evidence,
                                                        "retryable": True,
                                                        "next_action": (
                                                            "call final_answer again "
                                                            "with access_granted=true "
                                                            "and SQL rewritten to "
                                                            "authorized columns, or "
                                                            "deny with a header-grounded "
                                                            "unauthorized_column"
                                                        ),
                                                        "guidance": (
                                                            self._grant_grounding_guidance(
                                                                context
                                                            )
                                                        ),
                                                    }
                                                )
                                            ),
                                        )
                                    )
                                    continue
                                # Second ungrounded grant: fail closed to a formal
                                # denial that carries the offending citation.
                                tool_response, response, _sql = execute_final_answer(
                                    {
                                        "access_granted": False,
                                        "unauthorized_column": evidence,
                                    }
                                )
                                transcript.append(
                                    tool_result_message(
                                        tool_call,
                                        self._bound_tool_result(tool_response),
                                    )
                                )
                                return LLMCallResult(
                                    response=response or ACCESS_DENIED,
                                    messages=transcript,
                                    usage=total_usage,
                                    latency_seconds=time.perf_counter() - started,
                                    sql=None,
                                    turns=turns,
                                )
                        if (
                            requested_grant is False
                            and require_denial_evidence
                            and not denial_reconsider_used
                            and not named_unavailable_column(
                                policy_instruction, args.get("unauthorized_column")
                            )
                        ):
                            # The refusal names no column the structured header
                            # blocks. Ask once for header-grounded evidence (or a
                            # grant) instead of accepting a free denial.
                            denial_reconsider_used = True
                            transcript.append(
                                tool_result_message(
                                    tool_call,
                                    self._bound_tool_result(
                                        json.dumps(
                                            {
                                                "status": "not_accepted",
                                                "error": (
                                                    "a denial requires an "
                                                    "unauthorized_column that the "
                                                    "structured policy header blocks"
                                                ),
                                                "retryable": True,
                                                "next_action": (
                                                    "call final_answer again with "
                                                    "access_granted=false and a "
                                                    "header-grounded "
                                                    "unauthorized_column, or grant if "
                                                    "every required column is "
                                                    "accessible"
                                                ),
                                                "guidance": self._DENIAL_REMINDER,
                                            }
                                        )
                                    ),
                                )
                            )
                            continue
                        tool_response, response, sql = execute_final_answer(args)
                        transcript.append(
                            tool_result_message(
                                tool_call, self._bound_tool_result(tool_response)
                            )
                        )
                        if response is None:
                            needs_final_reminder = True
                            continue
                        sql_result = sql
                        return LLMCallResult(
                            response=response,
                            messages=transcript,
                            usage=total_usage,
                            latency_seconds=time.perf_counter() - started,
                            sql=sql_result,
                            turns=turns,
                        )
                    elif name == "test_sql":
                        sql_tests += 1
                        if sql_tests > self.max_sql_tests:
                            tool_response = json.dumps(
                                {
                                    "error": "test_sql verification budget exhausted",
                                    "retryable": False,
                                    "next_action": "call final_answer",
                                }
                            )
                        else:
                            tool_response = execute_test_sql(args, db_path=db_path)
                            submitted_sql = args.get("sql")
                            if (
                                recover_exhausted_grant
                                and isinstance(policy_instruction, str)
                                and isinstance(submitted_sql, str)
                                and submitted_sql.strip()
                                and not inaccessible_tables(
                                    policy_instruction, submitted_sql
                                )
                                and grant_is_grounded(
                                    policy_instruction, submitted_sql
                                )
                                is True
                            ):
                                # The model itself cited this SQL and it cleared
                                # every deterministic grant guard, so it is a
                                # bounded recovery candidate if the loop dies.
                                recovered_sql = submitted_sql
                        transcript.append(
                            tool_result_message(
                                tool_call, self._bound_tool_result(tool_response)
                            )
                        )
                        if sql_tests >= self.max_sql_tests:
                            needs_final_reminder = True
                    else:
                        transcript.append(
                            tool_result_message(
                                tool_call,
                                self._bound_tool_result(
                                    json.dumps({"error": f"unknown tool: {name}"})
                                ),
                            )
                        )

                # Tool results must remain adjacent to the assistant tool-call
                # message. Add any controlled reminder only after every result for
                # this turn has been recorded.
                if needs_final_reminder:
                    transcript.append({"role": "user", "content": self._FINAL_REMINDER})

            # Exhausted turns without final_answer.  Never raise: the required
            # 0/1 output contract must survive turn exhaustion.  Return the last
            # submission that already cleared both the whole-table veto and the
            # grant-side citation check, or fail closed to a formal denial when no
            # guard-cleared submission exists.
            if recovered_sql is not None:
                _payload, recovered_response, recovered_out = execute_final_answer(
                    {"access_granted": True, "sql": recovered_sql}
                )
                return LLMCallResult(
                    response=recovered_response or ACCESS_GRANTED,
                    messages=transcript,
                    usage=total_usage,
                    latency_seconds=time.perf_counter() - started,
                    sql=recovered_out,
                    turns=turns,
                )
            _payload, denied_response, _denied_sql = execute_final_answer(
                {"access_granted": False}
            )
            return LLMCallResult(
                response=denied_response or ACCESS_DENIED,
                messages=transcript,
                usage=total_usage,
                latency_seconds=time.perf_counter() - started,
                sql=None,
                turns=turns,
            )

        except Exception as exc:
            return LLMCallResult(
                response=ACCESS_DENIED,
                messages=transcript,
                usage=total_usage,
                latency_seconds=time.perf_counter() - started,
                sql=sql_result,
                turns=turns,
                error=f"{type(exc).__name__}: {exc}",
            )
