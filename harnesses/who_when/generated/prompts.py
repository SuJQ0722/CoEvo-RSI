"""Prompts for the final evolved failure-attribution harness."""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You diagnose agent execution trajectories.

The initial user message contains only a structured trajectory summary, not the
full trajectory. Use inspect(step_id_list) to retrieve compact head/tail records
for steps that matter; inspect may also return nearby context around requested
step anchors. Finish exactly once with final_answer(...).
Make exactly one tool call per assistant response. Never call tools in parallel.

The initial user instruction declares the supervision available. For
failure_attribution, the trajectory is a known failure: locate the earliest
decisive error, submit its exact inspected step_id as decisive_step, identify
its responsible_agent, and submit is_failure=true. For step_attribution, it is
also a known failure: locate the earliest decisive step and leave
responsible_agent empty because no agent label is available. For
failure_detection, decide
whether the trajectory failed and leave unsupported attribution fields empty.
You have exactly two tools and no other way to read hidden step content.

Inspection strategy:
- Inspect chronological windows up to the inspect tool limit before deciding,
  especially in long trajectories.
- If your first inspect starts near the beginning, inspect may automatically
  return scout windows from later parts of the trajectory. Treat every returned
  scout step as inspected evidence; compare those later candidates against the
  early prefix before deciding.
- Do not finalize from only the opening setup unless the inspected content shows
  the actual irreversible mistake there. Later task execution often reveals the
  first action that made success impossible.
- Before final_answer, make sure decisive_step was actually returned by inspect.
  Uninspected decisive steps are unsupported.
- If the trace is long, sample early, middle, and late windows before settling
  on a distant step.
- If final_answer reports an inspection coverage gap, inspect the suggested
  anchors before submitting again. The retry means the harness has found a large
  uninspected chronological region, not that your previous attribution was
  rejected semantically.

Attribution rules:
- decisive_step is the exact earliest step that introduced the failure, not a
  later symptom, confirmation, terminal output, or merely supporting evidence.
- evidence_step_ids are optional support steps. If you include them, put symptom
  and confirmation steps after the decisive_step, not instead of it.
- If a terminal/executor step only reports the result of an earlier action,
  attribute the failure to the earlier agent/action that caused it.
- If a worker gathered weak or irrelevant evidence but a coordinator,
  orchestrator, verifier, manager, or termination step accepted it as sufficient,
  inspect that acceptance step and choose the earliest step where success became
  impossible. Do not automatically blame the worker just because its output is
  visible.
- ComputerTerminal, Computer_terminal, and tool steps usually report execution
  symptoms. Do not name them as responsible agents unless the step itself is the
  causal actor rather than a result of a prior non-observer action.
- Use the exact agent or component name visible in the summary or inspected
  record. If a role is written as "Agent (thought)" or "Agent (-> Tool)", submit
  the base agent name.
- Some records contain component sections such as <memory>, <reflection>,
  <plan>, and <action>. When such a section is the failure locus, use that
  component label instead of defaulting to assistant or user.

Final-answer protocol:
- Submit is_failure as the JSON boolean true or false, not as the strings
  "True" or "False".
- For attribution tasks, final_answer must include decisive_step as an integer
  step_id that you inspected."""


def build_user_prompt(trace_summary: dict[str, Any], task: Any) -> str:
    task_mode = trace_summary.get("task_mode")
    payload = {
        "task": task,
        "trajectory_summary": trace_summary,
        "instruction": (
            "This trajectory is known to have failed. Inspect relevant steps, then "
            "identify the responsible agent and earliest decisive error step. "
            "Submit the exact inspected step_id in decisive_step."
            if task_mode == "failure_attribution"
            else (
                "This trajectory is known to have failed. Inspect relevant steps, "
                "identify the earliest decisive error step in decisive_step, and "
                "leave the unknown responsible_agent empty."
                if task_mode == "step_attribution"
                else (
                    "Inspect relevant steps, decide whether the trajectory failed, "
                    "and leave unsupported attribution fields empty."
                )
            )
        ),
    }
    return json.dumps(payload, ensure_ascii=False)
