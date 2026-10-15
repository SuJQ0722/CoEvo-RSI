"""Normalized trace objects used by the failure-detection harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_TEXT_FIELDS = ("action", "content", "thought", "observation")


@dataclass(frozen=True)
class TraceStep:
    step_id: int
    agent_id: str
    raw: dict[str, Any]

    @classmethod
    def from_record(cls, record: dict[str, Any], position: int) -> "TraceStep":
        step_id = record.get("step_id", record.get("step", position))
        try:
            step_id = int(step_id)
        except (TypeError, ValueError):
            step_id = position
        nested_agents: list[Any] = []
        for substep in record.get("substeps") or []:
            if not isinstance(substep, dict):
                continue
            nested_agent = (
                substep.get("agent_name")
                or substep.get("name")
                or substep.get("agent_id")
                or substep.get("role")
            )
            if nested_agent not in (None, "") and nested_agent not in nested_agents:
                nested_agents.append(nested_agent)
        agent = (
            record.get("agent_name")
            or record.get("name")
            or record.get("agent_id")
            or record.get("role")
            or (nested_agents[0] if len(nested_agents) == 1 else None)
            or "unknown"
        )
        return cls(
            step_id=step_id,
            agent_id=str(agent),
            raw={**record, "step_id": step_id},
        )

    def summary(self) -> dict[str, Any]:
        """Return routing information, never the full step payload."""
        result: dict[str, Any] = {
            "step_id": self.step_id,
            "agent": self.agent_id,
        }
        for key in ("role", "phase"):
            if self.raw.get(key) not in (None, ""):
                result[key] = self.raw[key]
        text_shape = {}
        for key in _TEXT_FIELDS:
            value = self.raw.get(key)
            if isinstance(value, str) and value:
                text_shape[key] = {
                    "characters": len(value),
                    "lines": value.count("\n") + 1,
                }
        if text_shape:
            result["text_shape"] = text_shape
        for key in ("input", "output"):
            value = self.raw.get(key)
            if isinstance(value, dict):
                result[f"{key}_fields"] = sorted(value)
        result["payload_fields"] = sorted(
            key
            for key, value in self.raw.items()
            if key not in {
                "step",
                "step_id",
                "agent_id",
                "agent_name",
                "name",
                "role",
                "phase",
            }
            and value not in (None, "", [], {})
        )
        return result


@dataclass
class AgentTrace:
    sample_id: str
    task: Any
    steps: list[TraceStep]
    is_failure: bool
    source_dataset: str = ""
    decisive_step: int | None = None
    responsible_agents: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "AgentTrace":
        steps = [
            TraceStep.from_record(step, position)
            for position, step in enumerate(record.get("trajectory", []), start=1)
            if isinstance(step, dict)
        ]
        decisive_step = record.get("decisive_step")
        try:
            decisive_step = int(decisive_step) if decisive_step is not None else None
        except (TypeError, ValueError):
            decisive_step = None
        responsible = record.get("responsible_agents") or []
        if not responsible and record.get("responsible_agent"):
            responsible = [record["responsible_agent"]]
        return cls(
            sample_id=str(record.get("sample_id") or record.get("source_id") or ""),
            task=record.get("task", ""),
            steps=steps,
            is_failure=bool(record.get("is_failure", False)),
            source_dataset=str(record.get("source_dataset", "")),
            decisive_step=decisive_step,
            responsible_agents=[str(agent) for agent in responsible],
            metadata=dict(record.get("metadata") or {}),
        )

    def structured_summary(self) -> dict[str, Any]:
        return {
            "task_mode": (
                "failure_attribution"
                if self.responsible_agents and self.decisive_step is not None
                else "step_attribution"
                if self.decisive_step is not None
                else "failure_detection"
            ),
            "num_steps": len(self.steps),
            "agents": sorted({step.agent_id for step in self.steps}),
            "steps": [step.summary() for step in self.steps],
        }

    def find_step(self, step_id: int) -> TraceStep | None:
        return next((step for step in self.steps if step.step_id == step_id), None)


@dataclass(frozen=True)
class DetectionResult:
    is_failure: bool
    evidence_step_ids: tuple[int, ...] = ()
    rationale: str = ""
    responsible_agent: str = ""
