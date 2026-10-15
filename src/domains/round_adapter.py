"""Domain-specific ports used by the shared harness/policy round orchestrator."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.domains.base import EvaluationResult
from src.domains.registry import get_domain_plugin
from src.domains.selection import quality_score
from src.meta_evolver.evolution import record_selection
from src.meta_evolver.feedback import build_failure_feedback, write_failure_feedback
from src.utils.persistence import atomic_json


@dataclass(frozen=True)
class EvaluationArtifact:
    """Uniform view over a domain's rollout and metric artifacts."""

    root: Path
    rollouts: Path
    metrics: Path


class RoundAdapter(Protocol):
    name: str
    harness_kind: str
    metric_names: tuple[str, ...]
    metric_scale_max: float
    test_is_derived: bool

    def add_arguments(self, parser: Any) -> None: ...
    def data_preparation_options(self, args: Any) -> dict[str, Any]: ...

    def python_executable(
        self, config: Mapping[str, Any], project_root: Path, fallback: str
    ) -> str: ...
    def prepare_data_commands(self, **kwargs: Any) -> list[tuple[list[str], str]]: ...
    def meta_evidence_source(self, round_root: Path, parent: EvaluationArtifact) -> Path: ...
    def artifact(self, root: Path, name: str) -> EvaluationArtifact: ...
    def external_artifact(self, directory: Path, name: str) -> EvaluationArtifact: ...
    def evaluation_commands(self, **kwargs: Any) -> list[tuple[list[str], str]]: ...
    def test_derivation_commands(self, **kwargs: Any) -> list[tuple[list[str], str]] | None: ...
    def artifact_complete(self, artifact: EvaluationArtifact, expected: int) -> bool: ...
    def validate_reuse(
        self, artifact: EvaluationArtifact, expected: int, **kwargs: Any
    ) -> tuple[bool, str]: ...
    def load_result(self, artifact: EvaluationArtifact) -> EvaluationResult: ...
    def mismatch_contract(self) -> Any: ...
    def select_candidate(self, **kwargs: Any) -> dict[str, Any]: ...
    def policy_validation(
        self, parent: Mapping[str, Any], child: Mapping[str, Any], tolerance: float
    ) -> tuple[bool, str]: ...
    def sampling_command(self, **kwargs: Any) -> list[str]: ...


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _complete_jsonl(path: Path, expected: int) -> bool:
    """Reject partial, malformed, or duplicate rollout artifacts."""
    if not path.is_file():
        return False
    sample_ids: list[str] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = str(record.get("sample_id") or "")
                if not sample_id:
                    return False
                sample_ids.append(sample_id)
    except (OSError, TypeError, json.JSONDecodeError):
        return False
    return len(sample_ids) == expected and len(set(sample_ids)) == expected


def _no_regression(
    metric_names: tuple[str, ...],
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
    tolerance: float,
) -> tuple[bool, str]:
    deltas = {name: float(child[name]) - float(parent[name]) for name in metric_names}
    excessive = {name: value for name, value in deltas.items() if value < -tolerance - 1e-12}
    if excessive:
        detail = ", ".join(f"{key}={value:+.4f}" for key, value in excessive.items())
        return False, f"validation regression exceeds {tolerance:g}: {detail}"
    return True, "trained child remains within configured metric tolerance"


def _common_sampling_command(**kwargs: Any) -> list[str]:
    command = [
        kwargs["python"],
        "-m",
        "src.training.pipeline",
        "--config",
        str(kwargs["config"]),
        "--data",
        str(kwargs["data"]),
        "--policy-model",
        kwargs["model"],
        "--base-url",
        kwargs["base_url"],
        "--harness",
        kwargs["harness"],
        "--attempts-per-item",
        str(kwargs["attempts"]),
        "--concurrency",
        str(kwargs["concurrency"]),
        "--quality-tier",
        str(kwargs["quality_tier"]),
        "--max-turns",
        str(kwargs["max_turns"]),
        "--output-dir",
        str(kwargs["output_dir"]),
        "--sampling-only",
    ]
    if kwargs.get("limit") is not None:
        command.extend(["--limit", str(kwargs["limit"])])
    if kwargs.get("resume"):
        command.append("--resume")
    return command


class ArtifactRoundAdapter:
    """Reusable artifacts and bookkeeping, with domain-supplied metric rules."""

    metric_scale_max = 1.0
    supports_lineage = False
    prefilter_requires_improvement = True
    supports_staged_selection = False

    def mismatch_contract(self):
        from src.diagnostics.mismatch import MismatchContract

        return MismatchContract()

    def add_arguments(self, parser):
        """Register optional domain CLI arguments."""

    def data_preparation_options(self, args):
        return {}

    def prepare_data_commands(self, **kwargs: Any) -> list[tuple[list[str], str]]:
        # Domains may override this to prepare their data.
        return []

    def evaluation_dependencies(self, config, data, project_root):
        """Additional immutable input identities beyond the primary task manifest."""
        return {}

    def meta_evidence_source(self, round_root: Path, parent: EvaluationArtifact) -> Path:
        return parent.root

    def artifact(self, root: Path, name: str) -> EvaluationArtifact:
        directory = root / name
        return EvaluationArtifact(
            directory, directory / "samples.jsonl", directory / "metrics.json"
        )

    def external_artifact(self, directory: Path, name: str) -> EvaluationArtifact:
        return EvaluationArtifact(
            directory, directory / "samples.jsonl", directory / "metrics.json"
        )

    def select_candidate(self, **kwargs: Any) -> dict[str, Any]:
        config = kwargs["config"]
        plugin = get_domain_plugin(config)
        parent_result = self.load_result(kwargs["parent_artifact"])
        child_result = self.load_result(kwargs["child_artifact"])
        decision = plugin.select(parent_result, child_result, config.get("selection") or {})
        name = "shortlist_child" if decision.accepted else "retain_parent"
        payload = {
            "decision": name,
            "active_after_selection": kwargs["child"] if decision.accepted else kwargs["parent"],
            "reason": decision.reason,
            "deltas": decision.deltas,
            "stages": [
                {
                    "stage": 1,
                    "parent_metrics": parent_result.metrics,
                    "child_metrics": child_result.metrics,
                    "parent_rollouts": str(kwargs["parent_artifact"].rollouts),
                    "child_rollouts": str(kwargs["child_artifact"].rollouts),
                }
            ],
        }
        if not decision.accepted:
            feedback_path = kwargs["report_path"].with_name("failure_feedback.json")
            feedback = build_failure_feedback(
                status="child_not_promoted",
                reason=decision.reason,
                causes=["domain_quality_gate_rejected_child"],
                evidence={"deltas": decision.deltas},
            )
            write_failure_feedback(feedback_path, feedback)
            payload["failure_feedback_path"] = str(feedback_path)
        atomic_json(kwargs["report_path"], payload)
        record_selection(
            kwargs["registry_path"],
            kwargs["parent"],
            kwargs["child"],
            "shortlist_child" if decision.accepted else "retain_parent",
            kwargs["report_path"],
        )
        return payload

    def quality_score(self, scores, selection):
        return quality_score(scores, selection)

    def policy_validation(
        self, parent: Mapping[str, Any], child: Mapping[str, Any], tolerance: float
    ) -> tuple[bool, str]:
        return _no_regression(self.metric_names, parent, child, tolerance)

    def sampling_command(self, **kwargs: Any) -> list[str]:
        return _common_sampling_command(**kwargs)


def get_round_adapter(config_or_name: Mapping[str, Any] | str) -> RoundAdapter:
    return get_domain_plugin(config_or_name).round_adapter()
