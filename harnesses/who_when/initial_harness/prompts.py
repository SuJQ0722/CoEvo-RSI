"""Prompts for the initial failure-detection harness."""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You diagnose agent execution trajectories.

The initial user message contains only a structured trajectory summary, not the
full trajectory. Use inspect(step_id_list) to retrieve full records for steps
that matter. Finish exactly once with final_answer(...).
Make exactly one tool call per assistant response. Never call tools in parallel.

The initial user instruction declares the supervision available. For
failure_attribution, the trajectory is a known failure: locate the earliest
decisive error, put its step in evidence_step_ids first, identify its
responsible_agent, and submit is_failure=true. For step_attribution, it is also
a known failure: locate the earliest decisive step and leave responsible_agent
empty because no agent label is available. For failure_detection, decide
whether the trajectory failed and leave unsupported attribution fields empty.
You have exactly two tools and no other way to read hidden step content."""


def build_user_prompt(trace_summary: dict[str, Any], task: Any) -> str:
    task_mode = trace_summary.get("task_mode")
    payload = {
        "task": task,
        "trajectory_summary": trace_summary,
        "instruction": (
            "This trajectory is known to have failed. Inspect relevant steps, then "
            "identify the responsible agent and earliest decisive error step."
            if task_mode == "failure_attribution"
            else (
                "This trajectory is known to have failed. Inspect relevant steps, "
                "identify the earliest decisive error step, and leave the unknown "
                "responsible_agent empty."
                if task_mode == "step_attribution"
                else (
                    "Inspect relevant steps, decide whether the trajectory failed, "
                    "and leave unsupported attribution fields empty."
                )
            )
        ),
    }
    return json.dumps(payload, ensure_ascii=False)
