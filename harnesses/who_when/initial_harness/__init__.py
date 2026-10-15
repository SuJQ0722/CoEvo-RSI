"""Initial harness — the seed. Tools are the harness implementation."""

from __future__ import annotations

import json
from typing import Any

from .tools.inspect import (
    SCHEMA as INSPECT_SCHEMA,
    execute as run_inspect,
)
from .tools.final_answer import (
    SCHEMA as FINAL_ANSWER_SCHEMA,
    execute as run_final_answer,
    schema_for_task as final_answer_schema_for_task,
)
from .prompts import SYSTEM_PROMPT, build_user_prompt
from src.traces import AgentTrace, DetectionResult


TOOLS = [INSPECT_SCHEMA, FINAL_ANSWER_SCHEMA]


def tools_for_trace(trace: AgentTrace) -> list[dict[str, Any]]:
    task_mode = trace.structured_summary()["task_mode"]
    return [INSPECT_SCHEMA, final_answer_schema_for_task(task_mode)]


class FailureDetectionHarness:
    name = "initial_harness"
    max_turns = 10

    def __init__(self, trace: AgentTrace):
        self.trace = trace
        self.task_mode = str(trace.structured_summary()["task_mode"])
        self._result: DetectionResult | None = None
        self.inspected_step_ids: list[int] = []

    @property
    def tools(self) -> list[dict[str, Any]]:
        return tools_for_trace(self.trace)

    def initial_messages(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    self.trace.structured_summary(), self.trace.task
                ),
            },
        ]

    def handle_tool_call(self, name: str, arguments: dict[str, Any]) -> str:
        if self.is_done:
            return json.dumps({"error": "final_answer already submitted"})
        if name == "inspect":
            return run_inspect(
                self.trace,
                arguments.get("step_id_list", []),
                self.inspected_step_ids,
            )
        if name == "final_answer":
            response, result = run_final_answer(arguments, task_mode=self.task_mode)
            if result is not None:
                self._result = result
            return response
        return json.dumps({"error": f"unknown tool: {name}"})

    @property
    def is_done(self) -> bool:
        return self._result is not None

    @property
    def result(self) -> DetectionResult | None:
        return self._result


def build_harness(trace: AgentTrace) -> FailureDetectionHarness:
    return FailureDetectionHarness(trace)
