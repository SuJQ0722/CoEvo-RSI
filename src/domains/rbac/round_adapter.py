"""Domain-owned round orchestration adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.domains.base import EvaluationResult
from src.domains.registry import get_domain_plugin
from src.domains.round_adapter import (
    ArtifactRoundAdapter,
    EvaluationArtifact,
    _common_sampling_command,
    _complete_jsonl,
    _json,
)
from src.meta_evolver.evolution import record_selection
from src.meta_evolver.feedback import build_failure_feedback, write_failure_feedback
from src.utils.persistence import atomic_json

from .evaluator import METRIC_NAMES as RBAC_METRIC_NAMES
from .evaluator import PROTOCOL as RBAC_PROTOCOL


class RBACRoundAdapter(ArtifactRoundAdapter):
    """Adapter for six-category, rule-first generative RBAC evolution."""

    name = "rbac"
    harness_kind = "rbac"
    metric_names = RBAC_METRIC_NAMES
    metric_scale_max = 1.0
    test_is_derived = False
    supports_lineage = True

    def mismatch_contract(self):
        from .mismatch import RBACMismatch

        return RBACMismatch()

    def evaluation_dependencies(self, config, data, project_root):
        from src.domains.io import read_records
        from src.meta_evolver.weights import file_hash

        from .databases import resolve_db_path

        result = {}
        for db_id in {str(row['db_id']) for row in read_records(data) if row.get('db_id')}:
            path = resolve_db_path(None, db_id, project_root=project_root)
            result[path or f'missing:{db_id}'] = file_hash(Path(path)) if path else 'missing'
        return result

    def python_executable(
        self, config: Mapping[str, Any], project_root: Path, fallback: str
    ) -> str:
        """Use the pipeline interpreter unless an environment is explicitly set."""
        prefix_value = (config.get("execution") or {}).get("environment_prefix")
        if not prefix_value:
            return fallback
        prefix = Path(str(prefix_value)).expanduser()
        if not prefix.is_absolute():
            prefix = project_root / prefix
        # Do not resolve the entry point: doing so can bypass the environment's
        # pyvenv/conda package context when bin/python is a symlink.
        python = prefix / "bin" / "python"
        if not python.is_file():
            raise FileNotFoundError(f"RBAC Python environment is missing: {python}")
        return str(python)

    def prepare_data_commands(self, **kwargs: Any) -> list[tuple[list[str], str]]:
        return []

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

    def evaluation_commands(
        self,
        *,
        python: str,
        config: Path,
        data: Path,
        split: str,
        model: str,
        base_url: str,
        harness: str,
        max_turns: int,
        concurrency: int,
        artifact: EvaluationArtifact,
        limit: int | None,
        resume: bool,
    ) -> list[tuple[list[str], str]]:
        del max_turns
        infer = [
            python,
            "-m",
            "src.domains.rbac.runtime.infer",
            "--config",
            str(config),
            "--data",
            str(data),
            "--model",
            model,
            "--base-url",
            base_url,
            "--harness",
            harness,
            "--concurrency",
            str(concurrency),
            "--output",
            str(artifact.rollouts),
        ]
        if limit is not None:
            infer.extend(["--limit", str(limit)])
        if resume:
            infer.append("--resume")
        evaluate = [
            python,
            "-m",
            "src.domains.rbac.runtime.evaluation",
            "--predictions",
            str(artifact.rollouts),
            "--output",
            str(artifact.metrics),
            "--harness",
            harness,
        ]
        return [(infer, f"{split}.rollout"), (evaluate, f"{split}.metrics")]

    def test_derivation_commands(self, **kwargs: Any) -> list[tuple[list[str], str]] | None:
        return None

    def artifact_complete(self, artifact: EvaluationArtifact, expected: int) -> bool:
        try:
            payload = _json(artifact.metrics)
            return (
                _complete_jsonl(artifact.rollouts, expected)
                and payload.get("protocol") == RBAC_PROTOCOL
                and int(payload.get("total", -1)) == expected
                and all(payload.get(name) is not None for name in self.metric_names)
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def validate_reuse(
        self, artifact: EvaluationArtifact, expected: int, **kwargs: Any
    ) -> tuple[bool, str]:
        if not self.artifact_complete(artifact, expected):
            return False, "RBAC rollout or metric artifact is incomplete"
        actual = _json(artifact.metrics).get("harness")
        expected_harness = str(kwargs["harness"])
        if actual != expected_harness:
            return False, f"artifact harness is {actual!r}, not {expected_harness!r}"
        return True, "RBAC artifact is complete and uses the requested harness"

    def load_result(self, artifact: EvaluationArtifact) -> EvaluationResult:
        payload = _json(artifact.metrics)
        return EvaluationResult(
            {name: float(payload[name]) for name in self.metric_names},
            per_sample=list(payload.get("per_sample") or []),
            diagnostics={
                key: payload.get(key)
                for key in (
                    "protocol",
                    "total",
                    "six_category_counts",
                    "confusion",
                    "precision",
                    "recall",
                    "per_database",
                    "per_dataset",
                    "per_database_categories",
                    "per_difficulty",
                    "per_difficulty_categories",
                    "protocol_invalid",
                    "sql_eval_available",
                )
            },
        )

    def select_candidate(self, **kwargs: Any) -> dict[str, Any]:
        plugin = get_domain_plugin(kwargs["config"])
        parent_result = self.load_result(kwargs["parent_artifact"])
        child_result = self.load_result(kwargs["child_artifact"])
        decision = plugin.select(
            parent_result,
            child_result,
            kwargs["config"].get("selection") or {},
        )
        action = "shortlist_child" if decision.accepted else "retain_parent"
        payload = {
            "decision": action,
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
            write_failure_feedback(
                feedback_path,
                build_failure_feedback(
                    status="child_not_promoted",
                    reason=decision.reason,
                    causes=["rbac_safe_ex_ac_f1_or_efficiency_gate_rejected_child"],
                    evidence={"deltas": decision.deltas},
                ),
            )
            payload["failure_feedback_path"] = str(feedback_path)
        atomic_json(kwargs["report_path"], payload)
        record_selection(
            kwargs["registry_path"],
            kwargs["parent"],
            kwargs["child"],
            action,
            kwargs["report_path"],
        )
        return payload

    def policy_validation(
        self, parent: Mapping[str, Any], child: Mapping[str, Any], tolerance: float
    ) -> tuple[bool, str]:
        safe_ex_delta = float(child["safe_ex"]) - float(parent["safe_ex"])
        ac_f1_delta = float(child["ac_f1"]) - float(parent["ac_f1"])
        if safe_ex_delta < -tolerance - 1e-12:
            return False, f"trained RBAC policy SafeEX regressed by {safe_ex_delta:+.6f}"
        if ac_f1_delta < -tolerance - 1e-12:
            return False, f"trained RBAC policy AC-F1 regressed by {ac_f1_delta:+.6f}"
        return True, "trained RBAC policy passes SafeEX and AC-F1 non-regression gates"

    def sampling_command(self, **kwargs: Any) -> list[str]:
        return _common_sampling_command(**kwargs)
