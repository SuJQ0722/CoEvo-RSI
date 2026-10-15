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

from .policy_guard import inaccessible_tables
from .tools.final_answer import ACCESS_DENIED, SCHEMA as FINAL_ANSWER_SCHEMA, execute as execute_final_answer
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

    def __init__(self, max_turns: int = 10, max_sql_tests: int = 8):
        self.max_turns = max_turns
        self.max_sql_tests = max_sql_tests

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

    def call(
        self,
        model: Any,
        messages: list[dict[str, str]],
        *,
        db_path: str | None = None,
        policy_instruction: str | None = None,
    ) -> LLMCallResult:
        started = time.perf_counter()
        transcript = list(messages)
        total_usage: dict[str, int] = {}
        sql_result: str | None = None
        turns = 0
        sql_tests = 0
        try:
            for turn in range(1, self.max_turns + 1):
                turns = turn
                assistant = dict(model.complete(transcript, self.tools))
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
                                tool_result_message(tool_call, tool_response)
                            )
                            return LLMCallResult(
                                response=response or ACCESS_DENIED,
                                messages=transcript,
                                usage=total_usage,
                                latency_seconds=time.perf_counter() - started,
                                sql=None,
                                turns=turns,
                            )
                        tool_response, response, sql = execute_final_answer(args)
                        transcript.append(tool_result_message(tool_call, tool_response))
                        if response is None:
                            raise ValueError("final_answer requires boolean access_granted")
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
                        transcript.append(tool_result_message(tool_call, tool_response))
                        if sql_tests >= self.max_sql_tests:
                            needs_final_reminder = True
                    else:
                        transcript.append(tool_result_message(
                            tool_call,
                            json.dumps({"error": f"unknown tool: {name}"}),
                        ))

                # Tool results must remain adjacent to the assistant tool-call
                # message. Add any controlled reminder only after every result for
                # this turn has been recorded.
                if needs_final_reminder:
                    transcript.append({"role": "user", "content": self._FINAL_REMINDER})

            # Exhausted turns without final_answer
            raise ValueError(f"exhausted {self.max_turns} turns without calling final_answer")

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
