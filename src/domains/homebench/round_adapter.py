"""HomeBench implementation of the generic round ports."""

import json
import math

from src.domains.base import EvaluationResult
from src.domains.round_adapter import ArtifactRoundAdapter, _complete_jsonl

from .data import PROTOCOL, load
from .evaluator import METRIC_NAMES
from .submission import SCORING_VERSION


class HomeBenchRoundAdapter(ArtifactRoundAdapter):
    name = harness_kind = "homebench"
    metric_names = METRIC_NAMES
    test_is_derived = False
    supports_lineage = True

    def mismatch_contract(self):
        from .mismatch import HomeBenchMismatch

        return HomeBenchMismatch()

    def evaluation_dependencies(self, config, data, project_root):
        from src.meta_evolver.weights import file_hash

        homes = data.parent / "homes.jsonl"
        # The portable export embeds home_status/methods in every task row, so
        # its primary split hash already freezes all environment inputs.
        return {str(homes): file_hash(homes)} if homes.is_file() else {}

    def python_executable(self, config, project_root, fallback):
        return fallback

    def evaluation_commands(
        self,
        *,
        python,
        config,
        data,
        split,
        model,
        base_url,
        harness,
        max_turns,
        concurrency,
        artifact,
        limit,
        resume,
    ):
        command = [
            python,
            "-m",
            "src.domains.homebench.runtime.infer",
            "--config",
            str(config),
            "--data",
            str(data),
            "--output-dir",
            str(artifact.root),
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
        ]
        if limit is not None:
            command += ["--limit", str(limit)]
        if resume:
            command.append("--resume")
        return [(command, f"{split}.homebench")]

    def test_derivation_commands(self, **kwargs):
        return None

    def artifact_complete(self, artifact, expected):
        try:
            payload = json.loads(artifact.metrics.read_text())
            from src.domains.io import read_records

            records = list(read_records(artifact.rollouts))
            bindings = payload["run_manifest"]["sample_digests"]
            ids = [r["sample_id"] for r in payload["results"]]
            return (
                _complete_jsonl(artifact.rollouts, expected)
                and payload["protocol"] == PROTOCOL
                and payload.get("scoring_version") == SCORING_VERSION
                and payload["sample_count"] == payload["num_results"] == expected
                and payload["infrastructure_errors"] == 0
                and len(ids) == len(set(ids)) == expected
                and set(ids) == set(bindings) == {r["sample_id"] for r in records}
                and all(
                    r.get("task_digest") == bindings[r["sample_id"]]
                    for r in records + payload["results"]
                )
                and all(math.isfinite(payload[k]) and 0 <= payload[k] <= 1 for k in METRIC_NAMES)
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def validate_reuse(self, artifact, expected, **kwargs):
        if not self.artifact_complete(artifact, expected):
            return False, "HomeBench artifact incomplete or contains infrastructure failures"
        from .runtime.infer import identity

        config = kwargs.get("config")
        if config is None or kwargs.get("data") is None:
            return False, "config and data are required to verify provenance"
        expected_identity = identity(
            list(load(kwargs["data"], limit=kwargs.get("limit"))),
            config,
            kwargs["harness"],
            model=kwargs["model"],
            base_url=kwargs["base_url"],
            max_turns=kwargs["max_turns"],
        )
        actual = json.loads(artifact.metrics.read_text())["run_manifest"]
        ok = actual == expected_identity
        return ok, "matching HomeBench provenance" if ok else "HomeBench provenance mismatch"

    def load_result(self, artifact):
        payload = json.loads(artifact.metrics.read_text())
        if payload.get("scoring_version") != SCORING_VERSION:
            raise ValueError("stale HomeBench scoring; rescore with the submission adapter")
        return EvaluationResult(
            {k: payload[k] for k in METRIC_NAMES},
            payload["results"],
            {
                k: payload[k]
                for k in ("protocol", "dataset_digest", "infrastructure_errors", "per_type")
            },
        )

    def policy_validation(self, parent, child, tolerance):
        ok = all(
            math.isfinite(float(child[k]))
            and float(child[k]) >= float(parent[k]) - tolerance - 1e-12
            for k in ("em", "f1")
        )
        return ok, "EM/F1 non-regression" if ok else "EM/F1 regressed or invalid"
