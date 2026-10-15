"""Architectural boundaries and extension tests for the reusable orchestration core."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.domains.base import BaseDomainPlugin, EvaluationResult, SelectionDecision, SmokeReport
from src.domains.registry import register_domain
from src.domains.round_adapter import ArtifactRoundAdapter
from src.training.sampling import SamplingStats, accepted_record, write_accepted_jsonl

ROOT = Path(__file__).resolve().parents[1]
# These are benchmark-specific command entrypoints, not reusable orchestration.
# Keep the shared runner modules under the strict domain-independence check.
DOMAIN_ENTRYPOINTS: set[str] = set()
CORE_FILES = [
    *sorted((ROOT / "src/training").glob("*.py")),
    *sorted((ROOT / "src/meta_evolver").glob("*.py")),
    *sorted(
        path for path in (ROOT / "src/runner").glob("*.py")
        if path.name not in DOMAIN_ENTRYPOINTS
    ),
    ROOT / "src/domains/round_adapter.py",
    ROOT / "scripts/run_one_round.py",
]


def test_core_does_not_import_concrete_domains_or_encode_attribution_fields():
    forbidden = (
        "harnesses.",
        "src.domains.who_when.runtime",
        *(f"src.domains.{name}" for name in ("who_when", "rbac", "homebench")),
    )
    for path in CORE_FILES:
        for node in ast.walk(ast.parse(path.read_text())):
            imports = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(name.startswith(forbidden) for name in imports), path
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in {
                    "agent_accuracy",
                    "step_accuracy",
                    "decisive_step",
                    "responsible_agent",
                    "who_when",
                    "rbac",
                }, path


def test_importing_orchestration_does_not_load_a_builtin_domain():
    script = """
import sys
import src.training.pipeline
import src.meta_evolver.runner
import src.runner.infer
import scripts.run_one_round
names = [name for name in sys.modules if name.startswith(('src.domains.who_when', 'src.domains.rbac', 'src.domains.homebench', 'harnesses.'))]
assert not names, names
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True, capture_output=True)


class ExtraAdapter(ArtifactRoundAdapter):
    name = "test_extra_domain"
    harness_kind = "custom_workflow"
    metric_names = ("utility",)
    test_is_derived = False

    def load_result(self, artifact):
        return EvaluationResult(json.loads(artifact.metrics.read_text()))


class ExtraDomain(BaseDomainPlugin):
    name = "test_extra_domain"
    harness_kind = "custom_workflow"

    def __init__(self):
        self.requests = []
        self.prepared = []

    def round_adapter(self):
        return ExtraAdapter()

    def collect_training(self, request):
        self.requests.append(request)
        record = accepted_record(
            sample_id="extra:1",
            messages=[
                {"role": "system", "content": "Complete this custom task."},
                {"role": "user", "content": "Return 42."},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "The requested value is 42.",
                    "tool_calls": [
                        {
                            "id": "a",
                            "type": "function",
                            "function": {
                                "name": "submit_custom_result",
                                "arguments": '{"value":42}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "submit_custom_result",
                    "tool_call_id": "a",
                    "content": '{"ok":true}',
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "submit_custom_result",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "integer"}},
                        },
                    },
                }
            ],
            sampling={"quality_tier": "strict"},
        )
        write_accepted_jsonl([record], request.output_dir / "accepted_rollouts.jsonl")
        return SamplingStats(items=1, attempts=1, accepted=1, strict_accepted=1)

    def prepare_training_source(self, request):
        self.prepared.append(request)
        return request.accepted_path

    def select(self, parent, child, config):
        delta = child.metrics["utility"] - parent.metrics["utility"]
        return SelectionDecision(delta > 0, "custom utility comparison", {"utility": delta})


@pytest.fixture
def extra_domain(tmp_path, monkeypatch):
    from src.domains import registry

    monkeypatch.setattr(registry, "_FACTORIES", dict(registry._FACTORIES))
    plugin = ExtraDomain()
    register_domain(plugin.name, lambda: plugin)
    harness = tmp_path / "harnesses/initial_harness"
    harness.mkdir(parents=True)
    data = tmp_path / "train.jsonl"
    data.write_text('{"sample_id":"extra:1"}\n')
    config = {
        "project": {"domain": plugin.name},
        "harness": {
            "kind": plugin.harness_kind,
            "package": "harnesses.test_extra_domain",
            "root": str(harness.parent),
            "base": "initial_harness",
            "active": "initial_harness",
            "max_turns": 7,
        },
        "selection": {"quality_metrics": ["utility"]},
        "rejection_sampling": {"quality_tier": "strict", "attempts_per_item": 2},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return plugin, path, data, harness


def test_new_domain_uses_shared_sampler_preparation_and_sft_without_core_edits(
    extra_domain, tmp_path, monkeypatch
):
    from src.training import pipeline

    plugin, config, data, _ = extra_domain
    launched = []
    monkeypatch.setattr(pipeline, "validate_ms_swift_environment", lambda **kwargs: None)
    monkeypatch.setattr(
        pipeline, "build_ms_swift_sft_command", lambda **kwargs: ["fake_sft", str(kwargs["data"])]
    )
    monkeypatch.setattr(
        pipeline.subprocess, "run", lambda command, **kwargs: launched.append(command)
    )
    monkeypatch.setattr(pipeline, "latest_ms_swift_checkpoint", lambda path: path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline",
            "--config",
            str(config),
            "--data",
            str(data),
            "--policy-model",
            "custom-policy",
            "--temperature",
            "0.3",
            "--output-dir",
            str(tmp_path / "training"),
        ],
    )
    pipeline.main()
    assert plugin.requests[0].max_turns == 7
    assert plugin.requests[0].attempts_per_item == 2
    assert plugin.requests[0].temperature == 0.3
    assert len(plugin.prepared) == len(launched) == 1
    prepared = json.loads(Path(launched[0][1]).read_text())
    assert prepared["tools"][0]["function"]["name"] == "submit_custom_result"
    assert "<think>" in prepared["messages"][2]["content"]


def test_new_domain_uses_shared_selection_and_lineage(extra_domain, tmp_path):
    from src.meta_evolver.evolution import (
        ensure_parent_harness_registered,
        load_harness_tree,
        register_candidate,
    )
    from src.meta_evolver.select import select_candidate

    plugin, config, _, harness = extra_domain
    registry = tmp_path / "registry.json"
    ensure_parent_harness_registered(registry, "initial_harness", harness)
    register_candidate(
        registry,
        "candidate",
        "initial_harness",
        tmp_path / "candidate",
        SmokeReport("candidate", True, ()),
    )
    adapter = plugin.round_adapter()
    artifacts = [adapter.artifact(tmp_path, name) for name in ("parent", "child")]
    for artifact, utility in zip(artifacts, (0.1, 0.8)):
        artifact.root.mkdir()
        artifact.metrics.write_text(json.dumps({"utility": utility}))
        artifact.rollouts.write_text('{"sample_id":"extra:1"}\n')
    report = select_candidate(
        config_path=config,
        registry_path=registry,
        parent_name="initial_harness",
        child_name="candidate",
        stages=[tuple(a.rollouts for a in artifacts)],
        report_path=tmp_path / "selection.json",
    )
    assert report["decision"] == "shortlist_child"
    assert report["reason"] == "custom utility comparison"
    assert (
        load_harness_tree(registry)["nodes"]["initial_harness"]["path"]
        != "harnesses/who_when/initial_harness"
    )


def test_missing_domain_does_not_silently_select_who_when():
    from src.domains.registry import get_domain_plugin
    from src.meta_evolver.evolution import load_harness_tree

    with pytest.raises(ValueError, match="explicitly"):
        get_domain_plugin({})
    assert (
        load_harness_tree(Path("/nonexistent/unused-domain-registry.json"))["nodes"][
            "initial_harness"
        ]["path"]
        is None
    )


@pytest.mark.parametrize("domain", ["who_when", "rbac", "homebench"])
def test_sampling_request_dispatch_preserves_domain_specific_budgets(domain, tmp_path, monkeypatch):
    from importlib import import_module

    from src.domains.base import SamplingRequest
    from src.training.rejection_sampling import collect_to_directory

    module = import_module(f"src.domains.{domain}.rejection_sampling")
    captured = []
    sentinel = SamplingStats(accepted=3)
    monkeypatch.setattr(module, "collect", lambda request: captured.append(request) or sentinel)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"project": {"domain": domain}}))
    request = SamplingRequest(
        config,
        tmp_path / "data.jsonl",
        tmp_path,
        "policy",
        "unused",
        "initial_harness",
        3,
        1,
        2,
        25,
        "strict",
    )
    assert collect_to_directory(request) is sentinel
    assert captured == [request]


@pytest.mark.parametrize("domain", ["who_when", "rbac", "homebench"])
@pytest.mark.parametrize("harness_mode", ["agent", "predefined"])
@pytest.mark.parametrize("model_mode", ["agent", "predefined"])
def test_shared_round_dry_run_uses_each_domain_without_schema_assumptions(
    domain, harness_mode, model_mode, tmp_path
):
    from src.config import load_config

    config = load_config(ROOT / "configs" / f"{domain}.yaml")
    config["inheritance"]["harness"]["mode"] = harness_mode
    config["inheritance"]["model"]["mode"] = model_mode
    # Dry-run fixtures exercise dispatch without requiring deployment-specific
    # dataset mounts, isolated action environments, GPU servers or API calls.
    data = tmp_path / "tasks.jsonl"
    data.write_text('{"sample_id":"dry-run-fixture"}\n')
    config["data"]["splits"] = dict.fromkeys(("train", "val", "test"), str(data))
    config.setdefault("execution", {})["environment_prefix"] = sys.prefix
    config["harness"]["active"] = "initial_harness"
    config["paths"]["output"] = str(tmp_path / "output")
    # Resolved configs may include credentials. Do not write them to fixtures.
    for provider in ("codex", "claude"):
        config.get("meta_evolver", {}).get(provider, {}).pop("api_key", None)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    process = subprocess.run(
        [
            sys.executable,
            "scripts/run_one_round.py",
            "--config",
            str(path),
            "--dry-run",
            "--max-harness-attempts",
            "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    preview = json.loads(process.stdout)
    assert preview["workflow"] == "LINEAGE"
    assert preview["validation"] == "full"
    assert preview["harness_shortlist"] == 3
    assert preview["third_harness"] == "within_non_policy_tolerances_of_second"
    assert preview["inheritance"]["harness"]["mode"] == harness_mode
    assert preview["inheritance"]["model"]["mode"] == model_mode
    assert preview["harness_collection"] == (
        "agent" if harness_mode == "agent" else "failure_similarity"
    )
    assert preview["model_traversal"] == (
        "agent_requested_full_val"
        if model_mode == "agent"
        else "all_compatible_singles_then_fixed_top_k_ties"
    )


@pytest.mark.parametrize("mode", ["fresh", "resume", "preaccepted"])
def test_all_training_entry_paths_apply_same_domain_curation_and_normalization(
    extra_domain, tmp_path, monkeypatch, mode
):
    from types import SimpleNamespace

    from src.meta_evolver.services import LocalRuntimeServices
    from src.training import pipeline
    from src.training.sampling import read_accepted_jsonl

    plugin, config_path, data, _ = extra_domain
    config = yaml.safe_load(config_path.read_text())
    config["data"] = {"splits": {"train": str(data)}}
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / mode
    accepted = output / "accepted_rollouts.jsonl"
    if mode != "fresh":
        plugin.collect_training(SimpleNamespace(output_dir=output))
        plugin.requests.clear()
    if mode == "resume":

        def resume(request):
            plugin.requests.append(request)
            return SamplingStats(skipped_existing=1, accepted=0)

        monkeypatch.setattr(plugin, "collect_training", resume)

    def curate(request):
        plugin.prepared.append(request)
        records = read_accepted_jsonl(request.accepted_path)
        for record in records:
            record["domain_curated"] = True
        path = request.output_dir / "curated.jsonl"
        write_accepted_jsonl(records, path)
        return path

    monkeypatch.setattr(plugin, "prepare_training_source", curate)
    launched = []
    monkeypatch.setattr(pipeline, "validate_ms_swift_environment", lambda **kwargs: None)
    monkeypatch.setattr(
        pipeline, "build_ms_swift_sft_command", lambda **kw: ["sft", str(kw["data"])]
    )
    monkeypatch.setattr(
        pipeline.subprocess, "run", lambda command, **kwargs: launched.append(command)
    )
    monkeypatch.setattr(pipeline, "latest_ms_swift_checkpoint", lambda path: path)
    args = [
        "pipeline",
        "--config",
        str(config_path),
        "--policy-model",
        "policy",
        "--output-dir",
        str(output),
    ]
    if mode == "preaccepted":
        args += ["--accepted-rollouts", str(accepted)]
    else:
        args += ["--data", str(data)]
    if mode == "resume":
        args += ["--resume"]
    monkeypatch.setattr(sys, "argv", args)
    pipeline.main()
    assert len(plugin.prepared) == len(launched) == 1
    assert len(plugin.requests) == (0 if mode == "preaccepted" else 1)
    if mode == "resume":
        assert plugin.requests[0].resume
    request = plugin.prepared[0]
    assert request.data_path == data
    prepared = Path(launched[0][1]).read_bytes()
    record = json.loads(prepared)
    assert json.loads(record["metadata_json"])["domain_curated"]
    assert "<think>" in record["messages"][2]["content"]
    assert record["tools"][0]["function"]["name"] == "submit_custom_result"
    services = LocalRuntimeServices(tmp_path, training_python=sys.executable)
    round_output = tmp_path / "round_training" / "sft_train.jsonl"
    count, _ = services.prepare_training_dataset(
        domain=plugin.name,
        quality_tier="strict",
        accepted_rollouts=accepted,
        training_dataset=round_output,
        train_data=data,
        harness_name="initial_harness",
        sampling_config=config["rejection_sampling"],
        training_config={},
    )
    assert count == 1 and round_output.read_bytes() == prepared


@pytest.mark.parametrize(
    "module",
    [
        "src.domains.who_when.runtime.infer",
        "src.domains.who_when.runtime.evaluation",
        "src.domains.who_when.runtime.build_dataset",
        "src.domains.rbac.runtime.infer",
        "src.domains.rbac.runtime.evaluation",
        "src.domains.homebench.runtime.infer",
    ],
)
def test_domain_runtime_cli_is_importable_after_migration(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize(
    "flag", ["--debug-data", "--selection-limit", "--test-limit", "--continue-candidate"]
)
def test_round_cli_rejects_retired_bypasses_before_execution(flag):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_one_round.py",
            "--config",
            "configs/who_when.yaml",
            flag,
            "1",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
