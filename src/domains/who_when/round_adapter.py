"""Domain-owned round orchestration adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.domains.base import EvaluationResult
from src.domains.round_adapter import (
    ArtifactRoundAdapter,
    EvaluationArtifact,
    _common_sampling_command,
    _complete_jsonl,
    _json,
    _no_regression,
)
from src.domains.selection import split_quality_scores


class WhoWhenRoundAdapter(ArtifactRoundAdapter):
    supports_staged_selection = True
    name = "who_when"
    harness_kind = "failure_detection"
    metric_names = ("agent_accuracy", "step_accuracy")
    metric_scale_max = 100.0
    test_is_derived = False
    supports_lineage = True
    prefilter_requires_improvement = False

    def mismatch_contract(self):
        from .mismatch import WhoWhenMismatch

        return WhoWhenMismatch()

    def add_arguments(self, parser):
        parser.add_argument("--source-data", type=Path)
        parser.add_argument("--who-and-when-root", type=Path)

    def data_preparation_options(self, args):
        return {
            "source_data_override": args.source_data,
            "who_and_when_root_override": args.who_and_when_root,
        }

    def python_executable(
        self, config: Mapping[str, Any], project_root: Path, fallback: str
    ) -> str:
        return fallback

    def prepare_data_commands(self, **kwargs: Any) -> list[tuple[list[str], str]]:
        py = kwargs["python"]
        project_root = kwargs["project_root"]
        paths = kwargs["paths"]

        def project_path(value: str | Path) -> Path:
            path = Path(value).expanduser()
            return path.resolve() if path.is_absolute() else (project_root / path).resolve()

        raw_source_data = project_path(
            kwargs.get("source_data_override")
            or paths.get("raw_source_data", "data/high_quality_training/merged.jsonl")
        )
        source_data = project_path(
            paths.get("source_data", "data/high_quality_training/merged_clean.jsonl")
        )
        who_and_when_root = project_path(
            kwargs.get("who_and_when_root_override")
            or paths.get("who_and_when_root", "../who_and_when/Who&When")
        )
        return [
            (
                [
                    py,
                    "-m",
                    "src.domains.who_when.runtime.build_dataset",
                    "--input",
                    str(raw_source_data),
                    "--output-dir",
                    str(kwargs["train_data"].parent),
                    "--clean-output",
                    str(source_data),
                    "--validation-count",
                    "200",
                    "--who-and-when",
                    str(who_and_when_root / "Algorithm-Generated"),
                    str(who_and_when_root / "Hand-Crafted"),
                ],
                "data.split",
            ),
        ]

    def meta_evidence_source(self, round_root: Path, parent: EvaluationArtifact) -> Path:
        return round_root

    def artifact(self, root: Path, name: str) -> EvaluationArtifact:
        stem = {
            "parent_val": "selection_parent",
            "child_val": "selection_child",
            "pre_sft_val": "pre_sft_validation",
            "policy_val": "policy_validation",
            "policy_test": "policy_test",
        }[name]
        return EvaluationArtifact(root, root / f"{stem}.jsonl", root / f"{stem}.metrics.json")

    def external_artifact(self, directory: Path, name: str) -> EvaluationArtifact:
        return self.artifact(directory, name)

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
        command = [
            python,
            "-m",
            "src.domains.who_when.runtime.infer",
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
            "--max-turns",
            str(max_turns),
            "--concurrency",
            str(concurrency),
            "--output",
            str(artifact.rollouts),
        ]
        if limit is not None:
            command.extend(["--limit", str(limit)])
        if resume:
            command.append("--resume")
        evaluate = [
            python,
            "-m",
            "src.domains.who_when.runtime.evaluation",
            "--predictions",
            str(artifact.rollouts),
            "--output",
            str(artifact.metrics),
        ]
        return [(command, f"{split}.rollout"), (evaluate, f"{split}.metrics")]

    def test_derivation_commands(self, **kwargs: Any) -> list[tuple[list[str], str]] | None:
        return None

    def artifact_complete(self, artifact: EvaluationArtifact, expected: int) -> bool:
        try:
            payload = _json(artifact.metrics)
            return (
                _complete_jsonl(artifact.rollouts, expected)
                and payload.get("protocol") == "who_and_when_official"
                and int(payload.get("total", -1)) == expected
                and all(payload.get(key) is not None for key in (*self.metric_names, "coverage"))
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def validate_reuse(
        self, artifact: EvaluationArtifact, expected: int, **kwargs: Any
    ) -> tuple[bool, str]:
        if not self.artifact_complete(artifact, expected):
            return False, "rollout or official metric artifact is incomplete"
        return True, "artifact is complete"

    def load_result(self, artifact: EvaluationArtifact) -> EvaluationResult:
        payload = _json(artifact.metrics)
        scores = {name: float(payload[name]) for name in self.metric_names}
        scores.update(split_quality_scores(payload, self.metric_names))
        return EvaluationResult(scores)

    def select_candidate(self, **kwargs: Any) -> dict[str, Any]:
        # Preserve the released Who&When staged-comparison implementation.
        from .selection import select_candidate

        return select_candidate(
            config_path=kwargs["config_path"],
            registry_path=kwargs["registry_path"],
            parent_name=kwargs["parent"],
            child_name=kwargs["child"],
            stages=kwargs.get("stages")
            or [(kwargs["parent_artifact"].rollouts, kwargs["child_artifact"].rollouts)],
            judge=kwargs.get("judge"),
            report_path=kwargs["report_path"],
        )

    def policy_validation(
        self, parent: Mapping[str, Any], child: Mapping[str, Any], tolerance: float
    ) -> tuple[bool, str]:
        return _no_regression(self.metric_names, parent, child, tolerance)

    def sampling_command(self, **kwargs: Any) -> list[str]:
        command = _common_sampling_command(**kwargs)
        command.extend(["--quality-tier", kwargs["quality_tier"]])
        return command
