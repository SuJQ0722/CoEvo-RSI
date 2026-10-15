"""Execute a model inside the fixed failure-detection harness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.diagnostics.observe import ObservedModel
from src.runner.model_client import ChatModel
from src.traces import AgentTrace, DetectionResult
from src.training.message_schema import (
    canonicalize_assistant_message,
    tool_result_message,
)
from harnesses.who_when.loader import build_harness


@dataclass
class Rollout:
    sample_id: str
    messages: list[dict[str, Any]]
    result: DetectionResult | None
    inspected_step_ids: list[int]
    turns: int
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    token_usage_available: bool = False
    max_context_chars: int = 0
    cumulative_context_chars: int = 0
    repeated_inspections: int = 0
    interface_trace: dict[str, Any] = field(default_factory=dict)

    def to_record(self, trace: AgentTrace) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_dataset": trace.source_dataset,
            "benchmark_split": trace.metadata.get("who_and_when_split"),
            "messages": self.messages,
            "prediction": (
                {
                    "is_failure": self.result.is_failure,
                    "evidence_step_ids": list(self.result.evidence_step_ids),
                    "rationale": self.result.rationale,
                    "responsible_agent": self.result.responsible_agent,
                }
                if self.result
                else None
            ),
            "gold": {
                "is_failure": trace.is_failure,
                "decisive_step": trace.decisive_step,
                "responsible_agent": trace.responsible_agents[0]
                if trace.responsible_agents
                else "",
            },
            "inspected_step_ids": self.inspected_step_ids,
            "turns": self.turns,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "token_usage_available": self.token_usage_available,
            "max_context_chars": self.max_context_chars,
            "cumulative_context_chars": self.cumulative_context_chars,
            "repeated_inspections": self.repeated_inspections,
            "error": self.error,
            "interface_trace": self.interface_trace,
        }


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    value = tool_call.get("function", {}).get("arguments", {})
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def run_rollout(
    trace: AgentTrace,
    model: ChatModel,
    max_turns: int = 10,
    harness_name: str = "initial_harness",
) -> Rollout:
    harness = build_harness(harness_name, trace)
    model = ObservedModel(model)
    messages = harness.initial_messages()
    error = ""
    termination = "unknown"
    turns = 0
    prompt_tokens = completion_tokens = total_tokens = 0
    token_usage_available = False
    max_context_chars = cumulative_context_chars = repeated_inspections = 0

    try:
        for turns in range(1, max_turns + 1):
            context_chars = len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))
            max_context_chars = max(max_context_chars, context_chars)
            cumulative_context_chars += context_chars
            assistant = canonicalize_assistant_message(
                dict(model.complete(messages, harness.tools))
            )
            usage = assistant.pop("_coevox_usage", {})
            if isinstance(usage, dict) and usage:
                token_usage_available = True
                prompt_tokens += int(usage.get("prompt_tokens") or 0)
                completion_tokens += int(usage.get("completion_tokens") or 0)
                total_tokens += int(usage.get("total_tokens") or 0)
            assistant.setdefault("role", "assistant")
            messages.append(assistant)
            tool_calls = assistant.get("tool_calls") or []
            if not tool_calls:
                error = "assistant returned no tool call"
                termination = "protocol_error"
                break
            for index, tool_call in enumerate(tool_calls):
                call_id = tool_call.get("id") or f"call_{turns}_{index}"
                tool_call["id"] = call_id
                name = tool_call.get("function", {}).get("name", "")
                if index == 0:
                    arguments = _arguments(tool_call)
                    requested_steps = arguments.get("step_id_list")
                    inspected_before = set(harness.inspected_step_ids)
                    if isinstance(requested_steps, list):
                        for value in requested_steps:
                            try:
                                repeated_inspections += int(int(value) in inspected_before)
                            except (TypeError, ValueError):
                                continue
                    result = harness.handle_tool_call(name, arguments)
                else:
                    result = json.dumps(
                        {
                            "error": (
                                "Only one tool call is allowed per assistant "
                                "response. Submit this call again in the next turn."
                            ),
                            "retryable": True,
                        }
                    )
                messages.append(tool_result_message(tool_call, result))
                if harness.is_done:
                    break
            if harness.is_done:
                termination = "submitted"
                break
        else:
            error = "max turns reached without final_answer"
            termination = "budget_exhausted"
    except Exception as exc:  # noqa: BLE001 -- preserve partial rollout for diagnosis
        error = f"{type(exc).__name__}: {exc}"
        termination = (
            "provider_error"
            if any("error_type" in request for request in model.requests)
            else "runtime_error"
        )

    return Rollout(
        sample_id=trace.sample_id,
        messages=messages,
        result=harness.result,
        inspected_step_ids=harness.inspected_step_ids,
        turns=turns,
        error=error,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        token_usage_available=token_usage_available,
        max_context_chars=max_context_chars,
        cumulative_context_chars=cumulative_context_chars,
        repeated_inspections=repeated_inspections,
        interface_trace=model.trace(termination),
    )
