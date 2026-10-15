import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import harnesses.who_when.candidates as generated_harnesses
from src.config import active_harness, load_config
from src.domains.round_adapter import get_round_adapter
from src.domains.who_when.runtime.evaluation import StrictAttributionJudge
from src.domains.who_when.selection import (
    HarnessMetrics,
    decide_selection,
    pareto_relation,
    regression_cases,
    select_candidate,
    smoke_test_harness,
    summarize_failed_child,
)
from src.meta_evolver.batch_summary import (
    build_batch_summary_prompt,
    summarize_failure_batch,
)
from src.meta_evolver.context import briefs_schema, parse_json_response
from src.meta_evolver.evolution import (
    load_harness_tree,
    register_candidate,
    semantic_harness_hash,
    validate_candidate_bundle,
)
from src.meta_evolver.prompts import build_prompt as build_meta_evolver_prompt
from src.meta_evolver.runner import (
    _analysis_report_violations,
    _latest_failure_feedback,
    _prepare_trajectory_directory,
    _subprocess_environment,
    build_cli_command,
    run_meta_evolver,
)
from src.meta_evolver.services import command_logger_name
from harnesses.who_when import loader as harness_loader


def test_parse_json_response_accepts_raw_json():
    assert parse_json_response('{"target": "raw"}') == {"target": "raw"}


def test_parse_json_response_migrates_legacy_fenced_json():
    response = """Analysis complete. The packet is below.

```json
{"target": "deepseek", "references": []}
```

Additional notes follow.
"""
    assert parse_json_response(response) == {
        "target": "deepseek",
        "references": [],
    }


def test_parse_json_response_extracts_json_before_trailing_commentary():
    response = '{"target": "deepseek", "references": []}\n\nSummary follows.'
    assert parse_json_response(response) == {
        "target": "deepseek",
        "references": [],
    }


def test_parse_json_response_rejects_malformed_json():
    try:
        parse_json_response("Result: ```json\n{not valid}\n```")
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("malformed JSON should not be repaired")


def test_briefs_output_schema_uses_provider_compatible_object_root():
    schema = briefs_schema(8)
    assert schema["type"] == "object"
    assert schema["properties"]["briefs"]["minItems"] == 8
    assert schema["properties"]["briefs"]["maxItems"] == 8


def test_config_resolves_api_key_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_CODEX_KEY", "secret-value")
    path = tmp_path / "config.yaml"
    path.write_text("api_key: ${TEST_CODEX_KEY}\n", encoding="utf-8")
    assert load_config(path)["api_key"] == "secret-value"


def test_config_loads_shell_style_project_dotenv_without_overriding_environment(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    configs = project / "configs"
    configs.mkdir(parents=True)
    (project / ".env").write_text(
        "export OPENAI_API_KEY=dotenv-key\nOPENAI_BASE_URL='https://example.invalid/v1'\n",
        encoding="utf-8",
    )
    path = configs / "base.yaml"
    path.write_text("key: ${OPENAI_API_KEY}\nurl: ${OPENAI_BASE_URL}\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://process.invalid/v1")
    loaded = load_config(path)
    assert loaded["key"] == "dotenv-key"
    assert loaded["url"] == "https://process.invalid/v1"


def test_active_harness_defaults_to_initial_harness():
    assert active_harness({}) == "initial_harness"
    assert active_harness({"harness": {"active": "candidate_1"}}) == "candidate_1"


def test_cli_commands_are_non_interactive_and_do_not_contain_api_keys(tmp_path):
    codex = build_cli_command(
        "codex",
        {
            "command": "codex",
            "api_key": "secret-value",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://example.invalid/v1",
            "model": "custom-codex-model",
            "sandbox": "workspace-write",
        },
        tmp_path,
        tmp_path / "final.md",
    )
    assert codex[:2] == ["codex", "exec"]
    assert "workspace-write" in codex
    assert 'approval_policy="never"' in codex
    assert "--skip-git-repo-check" in codex
    assert "--ignore-user-config" in codex
    assert "--strict-config" in codex
    assert codex[codex.index("--enable") + 1] == "multi_agent"
    assert "--search" not in codex
    assert 'model_reasoning_effort="medium"' in codex
    assert 'model_provider="coevox_custom"' in codex
    assert 'model_providers.coevox_custom.env_key="OPENAI_API_KEY"' in codex
    assert 'model_providers.coevox_custom.base_url="https://example.invalid/v1"' in codex
    assert codex[codex.index("--model") + 1] == "custom-codex-model"
    assert codex[-1] == "-"
    assert "secret-value" not in codex

    schema = tmp_path / "schema.json"
    structured = build_cli_command(
        "codex",
        {"command": "codex", "sandbox": "workspace-write"},
        tmp_path,
        tmp_path / "structured.json",
        output_schema_path=schema,
    )
    assert structured[structured.index("--output-schema") + 1] == str(schema)

    claude = build_cli_command(
        "claude",
        {"command": "claude", "permission_mode": "acceptEdits"},
        tmp_path,
        tmp_path / "final.md",
    )
    assert claude == ["claude", "-p", "--permission-mode", "acceptEdits"]


def test_round_pipeline_logs_the_executed_module():
    assert (
        command_logger_name(
            ["python", "-m", "src.domains.who_when.runtime.infer", "--config", "config.yaml"]
        )
        == "domains.who_when.runtime.infer"
    )
    assert (
        command_logger_name(["python", "-m", "src.meta_evolver.select"])
        == "meta_evolver.select"
    )
    assert command_logger_name(["vllm", "serve", "model"]) == "runner.vllm_server"


def test_trained_policy_gate_allows_regression_with_strict_mean_improvement():
    accepted, _ = get_round_adapter("who_when").policy_validation(
        {"agent_accuracy": 50.0, "step_accuracy": 50.0},
        {"agent_accuracy": 56.0, "step_accuracy": 45.0},
        5.0,
    )
    assert accepted


def test_trained_policy_gate_rejects_excessive_regression():
    accepted, reason = get_round_adapter("who_when").policy_validation(
        {"agent_accuracy": 50.0, "step_accuracy": 50.0},
        {"agent_accuracy": 60.0, "step_accuracy": 44.0},
        5.0,
    )
    assert not accepted
    assert "step_accuracy" in reason


def test_trained_policy_gate_ignores_coverage():
    accepted, _ = get_round_adapter("who_when").policy_validation(
        {"agent_accuracy": 40.0, "step_accuracy": 40.0, "coverage": 0.95},
        {"agent_accuracy": 46.0, "step_accuracy": 45.0, "coverage": 0.10},
        5.0,
    )
    assert accepted


def test_trained_policy_gate_accepts_equal_mean_within_tolerance():
    accepted, reason = get_round_adapter("who_when").policy_validation(
        {"agent_accuracy": 50.0, "step_accuracy": 50.0},
        {"agent_accuracy": 55.0, "step_accuracy": 45.0},
        10.0,
    )
    assert accepted
    assert "within configured metric tolerance" in reason


def test_semantic_harness_hash_ignores_comments_docstrings_and_identity(tmp_path):
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    (parent / "__init__.py").write_text(
        '"""parent docs"""\nclass Harness:\n    name = "parent"\n    def value(self):\n        return 1\n',
        encoding="utf-8",
    )
    (child / "__init__.py").write_text(
        '"""child docs"""\n# comment\nclass Harness:\n    name = "child"\n    def value(self):\n        """method docs"""\n        return 1  # same behavior\n',
        encoding="utf-8",
    )
    assert semantic_harness_hash(parent) == semantic_harness_hash(child)

    (child / "__init__.py").write_text(
        'class Harness:\n    name = "child"\n    def value(self):\n        return 2\n',
        encoding="utf-8",
    )
    assert semantic_harness_hash(parent) != semantic_harness_hash(child)


def test_semantic_harness_hash_ignores_unreachable_python(tmp_path):
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    for root in (parent, child):
        (root / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (child / "dead_code.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert semantic_harness_hash(parent) == semantic_harness_hash(child)


def test_codex_subprocess_uses_isolated_writable_home(tmp_path):
    codex_home = tmp_path / "codex_home"
    environment = _subprocess_environment({}, codex_home, tmp_path)
    assert environment["CODEX_HOME"] == str(codex_home)
    assert environment["PYTHONPATH"].split(":")[0] == str(tmp_path)
    assert codex_home.is_dir()


def test_latest_rejected_child_feedback_is_reused(tmp_path):
    report = tmp_path / "selection.json"
    report.write_text(
        json.dumps({"failure_feedback": {"failure_causes": ["lower_f1"]}}),
        encoding="utf-8",
    )
    registry = tmp_path / "tree.json"
    registry.write_text(
        json.dumps(
            {
                "nodes": {
                    "parent": {"parent": None},
                    "child": {
                        "parent": "parent",
                        "selections": [
                            {
                                "evaluated_at": "2026-01-01T00:00:00Z",
                                "decision": "retain_parent",
                                "report": str(report),
                            }
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    feedback = _latest_failure_feedback(registry, tmp_path, "parent")
    assert feedback == {"failure_causes": ["lower_f1"]}


def test_meta_prompt_combines_prior_analysis_and_failure_feedback(tmp_path):
    prompt = build_meta_evolver_prompt(
        workspace_root=tmp_path,
        evidence_path=tmp_path / "evidence.jsonl",
        candidate_name="child_retry",
        parent_name="parent",
        instruction="Improve failure detection.",
        max_trace_items=10,
        prior_feedback_path=tmp_path / "prior_failure_feedback.json",
        prior_analysis_path=tmp_path / "prior_analysis.md",
        domain_user_prompt="Primary validation metrics: score, completion_rate.",
    )
    assert "prior_failure_feedback.json" in prompt
    assert "prior_analysis.md" in prompt
    assert "Primary validation metrics: score, completion_rate" in prompt
    assert "Call spawn_agent exactly once" not in prompt
    assert "Only candidate/ may change" in prompt
    assert "harness-addressable" in prompt
    assert "model-capability-limited" in prompt
    assert "external/data" in prompt
    assert "3-6 item failure taxonomy" in prompt
    assert "HARNESS DIAGNOSIS" in prompt
    assert "source URLs" not in prompt
    assert "candidate/evolution_report.md" in prompt
    assert "literal http:// or https://" not in prompt


def test_candidate_evolution_report_with_literal_source_url_passes_boundary(tmp_path):
    report = tmp_path / "evolution_report.md"
    report.write_text(
        "Changed candidate/agent.py. "
        + "Evidence, expected metrics, validation, assumptions, and risks. " * 12
        + "Source: https://docs.python.org/3/library/pathlib.html\n",
        encoding="utf-8",
    )

    assert _analysis_report_violations(report) == []


def test_meta_prompt_does_not_require_external_research(tmp_path):
    prompt = build_meta_evolver_prompt(
        workspace_root=tmp_path,
        evidence_path=tmp_path / "evidence.jsonl",
        candidate_name="candidate_01",
        parent_name="initial_harness",
        instruction="Improve failures.",
        max_trace_items=None,
        allow_research_subagent=False,
    )

    assert "authoritative external source" not in prompt
    assert "Call spawn_agent exactly once" not in prompt


def test_meta_report_does_not_require_an_external_url(tmp_path):
    report = tmp_path / "final_message.md"
    report.write_text(
        "Changed candidate/harness.py. " + "Evidence-backed validation and risks. " * 20,
        encoding="utf-8",
    )
    assert _analysis_report_violations(report) == []


def test_meta_prompt_uses_compact_workflow_for_seeded_resume(tmp_path):
    prompt = build_meta_evolver_prompt(
        workspace_root=tmp_path,
        evidence_path=tmp_path / "evidence.jsonl",
        candidate_name="candidate_01",
        parent_name="initial_harness",
        instruction="Finish the seeded candidate.",
        max_trace_items=None,
        prior_feedback_path=tmp_path / "prior_failure_feedback.json",
        prior_analysis_path=tmp_path / "prior_analysis.md",
        allow_research_subagent=False,
        resume_seeded=True,
    )

    assert "Seeded-resume workflow" in prompt
    assert "Do not repeat the full evidence audit" in prompt
    assert "Reserve enough response budget for the final report" in prompt
    assert "HARNESS DIAGNOSIS` with baseline metrics" not in prompt
    assert "Research at least one authoritative external source directly" not in prompt


def test_codex_command_can_disable_multi_agent(tmp_path):
    command = build_cli_command(
        "codex",
        {
            "command": "codex",
            "sandbox": "workspace-write",
            "multi_agent": False,
        },
        tmp_path,
        tmp_path / "final.md",
    )

    assert "multi_agent" not in command


def test_batch_summary_prompt_is_metric_driven_and_summary_only(tmp_path):
    prompt = build_batch_summary_prompt(tmp_path / "batch.json")
    assert "Agent Accuracy" in prompt
    assert "Step Accuracy" in prompt
    assert "harness-addressable" in prompt
    assert "model-capability-limited" in prompt
    assert "NEXT BATCH GUIDANCE" in prompt
    assert "Do not edit any file" in prompt


def test_batch_summary_dry_run_builds_independent_metaagent_command(tmp_path):
    config = {
        "meta_evolver": {
            "backend": "codex",
            "codex": {"command": "codex", "model": "gpt-5.5"},
        }
    }
    summary_path, command = summarize_failure_batch(
        config=config,
        project_root=tmp_path,
        feedback_path=tmp_path / "batch.json",
        output_dir=tmp_path / "summary",
        dry_run=True,
    )
    assert summary_path == tmp_path / "summary/workspace/failure_summary.md"
    assert command[:2] == ("codex", "exec")


def test_codex_extra_args_cannot_widen_write_permissions(tmp_path):
    try:
        build_cli_command(
            "codex",
            {"command": "codex", "extra_args": ["--add-dir", str(tmp_path.parent)]},
            tmp_path,
            tmp_path / "final.md",
        )
    except ValueError as exc:
        assert "unsafe" in str(exc)
    else:
        raise AssertionError("unsafe --add-dir should be rejected")


def test_dry_run_uses_configured_backend_without_spawning(tmp_path):
    traces = tmp_path / "rollouts.jsonl"
    traces.write_text("{}\n", encoding="utf-8")
    (tmp_path / "harnesses/who_when/initial_harness").mkdir(parents=True)
    config = {
        "project": {"domain": "who_when"},
        "harness": {
            "kind": "failure_detection",
            "package": "harnesses.who_when",
            "root": "harnesses/who_when",
            "base": "initial_harness",
            "active": "initial_harness",
        },
        "meta_evolver": {
            "backend": "codex",
            "output_dir": "output/meta_evolver",
            "codex": {"command": "codex"},
        },
    }
    result = run_meta_evolver(config, tmp_path, traces, dry_run=True)
    assert result.backend == "codex"
    assert result.return_code == 0
    assert result.command[-1] == "-"
    assert result.parent_harness == "initial_harness"
    assert result.smoke_passed is None


def test_meta_evidence_stages_all_top_level_rollout_files_from_directory(tmp_path):
    source = tmp_path / "round"
    source.mkdir()
    for name in ("selection_parent.jsonl", "test_parent.jsonl"):
        (source / name).write_text(json.dumps({"sample_id": name}) + "\n", encoding="utf-8")
    (source / "selection_parent.metrics.json").write_text("{}\n", encoding="utf-8")

    staged = _prepare_trajectory_directory(source, tmp_path / "trajectories")

    assert staged == tmp_path / "trajectories"
    assert sorted(path.name for path in staged.iterdir()) == [
        "selection_parent.jsonl",
        "test_parent.jsonl",
    ]


def test_initial_harness_passes_evolution_smoke_test():
    report = smoke_test_harness("initial_harness")
    assert report.passed
    assert {check.name for check in report.checks} == {
        "import_and_build",
        "identity",
        "local_runtime_dependencies",
        "two_tool_contract",
        "summary_boundary",
        "inspect_execution",
        "final_answer_execution",
    }


def test_generated_harness_identity_is_bound_to_registered_name(monkeypatch):
    harness = SimpleNamespace(
        name="initial_harness",
        tools=[],
        initial_messages=list,
        handle_tool_call=lambda *_args: "{}",
        is_done=False,
        result=None,
        inspected_step_ids=[],
    )
    module = SimpleNamespace(build_harness=lambda _trace: harness)
    monkeypatch.setattr(harness_loader.importlib, "import_module", lambda _name: module)

    built = harness_loader.build_harness("candidate_01", object())

    assert built.name == "candidate_01"


def test_candidate_bundle_allows_local_prompt_definitions(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "__init__.py").write_text('LOCAL_PROMPT = "do something"\n', encoding="utf-8")
    violations = validate_candidate_bundle(candidate)
    assert not any("prompt" in v.lower() for v in violations)


def test_candidate_bundle_allows_runtime_shell_and_docker_assets(tmp_path):
    candidate = tmp_path / "candidate"
    (candidate / "scripts").mkdir(parents=True)
    (candidate / "images/da-agent").mkdir(parents=True)
    (candidate / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (candidate / "scripts/entrypoint.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
    )
    (candidate / "images/da-agent/Dockerfile").write_text("FROM scratch\n", encoding="utf-8")

    violations = validate_candidate_bundle(candidate)

    assert not any("unsupported candidate file" in value for value in violations)


def test_registry_records_parent_child_lineage(tmp_path):
    registry = tmp_path / "tree.json"
    smoke = smoke_test_harness("initial_harness")
    register_candidate(
        registry,
        "candidate_1",
        "initial_harness",
        Path("harnesses/who_when/candidates/candidate_1"),
        smoke,
    )
    tree = load_harness_tree(registry)
    assert tree["nodes"]["candidate_1"]["parent"] == "initial_harness"
    assert tree["nodes"]["candidate_1"]["generation"] == 1
    assert tree["active"] == "initial_harness"


def test_legacy_registry_exposes_current_initial_harness(tmp_path):
    registry = tmp_path / "tree.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "active": "legacy_child",
                "nodes": {
                    "system_zero": {"parent": None, "generation": 0},
                    "legacy_child": {"parent": "system_zero", "generation": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    tree = load_harness_tree(registry)

    assert tree["active"] == "legacy_child"
    assert tree["nodes"]["initial_harness"] == {
        "parent": None,
        "generation": 0,
        "status": "available",
        "path": None,
        "smoke": None,
        "selections": [],
    }


def test_harness_prefilter_uses_agent_and_step_accuracy_tolerance():
    parent = HarnessMetrics(80.0, 80.0, 1.0, 0.0, 1.0, 1.0, 10)
    better_child = HarnessMetrics(90.0, 90.0, 1.0, 0.0, 1.0, 1.0, 10)
    costly_child = HarnessMetrics(90.0, 90.0, 1.0, 0.0, 2.0, 1.0, 10)
    worse_child = HarnessMetrics(70.0, 70.0, 1.0, 0.0, 2.0, 1.0, 10)

    assert pareto_relation(parent, better_child) == "child_dominates"
    stage = lambda parent, child: [
        {
            "stage": 1,
            "parent_metrics": parent.to_dict(),
            "child_metrics": child.to_dict(),
        }
    ]
    assert (
        decide_selection(True, stage(parent, better_child), quality_tolerance=5.0)[0]
        == "shortlist_child"
    )
    assert pareto_relation(parent, costly_child) == "non_dominated"
    assert (
        decide_selection(True, stage(parent, costly_child), quality_tolerance=5.0)[0]
        == "shortlist_child"
    )
    assert pareto_relation(parent, worse_child) == "parent_dominates"
    assert (
        decide_selection(True, stage(parent, worse_child), quality_tolerance=5.0)[0]
        == "retain_parent"
    )
    assert (
        decide_selection(False, stage(parent, better_child), quality_tolerance=5.0)[0]
        == "retain_parent"
    )

    within_five_points = HarnessMetrics(75.0, 75.0, 0.1, 0.9, 9.0, 99.0, 10)
    assert (
        decide_selection(True, stage(parent, within_five_points), quality_tolerance=5.0)[0]
        == "shortlist_child"
    )


def test_failed_child_feedback_records_metric_regressions():
    feedback = summarize_failed_child(
        "parent dominates",
        [
            {
                "stage": 1,
                "relation": "parent_dominates",
                "parent_metrics": {
                    "agent_accuracy": 80.0,
                    "step_accuracy": 80.0,
                    "coverage": 1.0,
                    "error_rate": 0.0,
                    "avg_turns": 2.0,
                    "avg_inspected_steps": 1.0,
                },
                "child_metrics": {
                    "agent_accuracy": 70.0,
                    "step_accuracy": 70.0,
                    "coverage": 1.0,
                    "error_rate": 0.1,
                    "avg_turns": 3.0,
                    "avg_inspected_steps": 1.0,
                },
            }
        ],
        {"passed": True},
    )
    assert "lower_agent_accuracy" in feedback["failure_causes"]
    assert "lower_step_accuracy" in feedback["failure_causes"]
    assert "higher_error_rate" not in feedback["failure_causes"]


def test_regression_cases_capture_parent_correct_child_wrong():
    parent = [
        {
            "sample_id": "a",
            "gold": {"responsible_agent": "agent", "decisive_step": 1},
            "prediction": {"responsible_agent": "agent", "evidence_step_ids": [1]},
            "turns": 2,
            "inspected_step_ids": [1],
            "error": "",
        }
    ]
    child = [
        {
            "sample_id": "a",
            "gold": {"responsible_agent": "agent", "decisive_step": 1},
            "prediction": {"responsible_agent": "other", "evidence_step_ids": [2]},
            "turns": 3,
            "inspected_step_ids": [1, 2],
            "error": "",
        }
    ]
    cases = regression_cases(parent, child)
    assert cases[0]["sample_id"] == "a"
    assert cases[0]["turn_delta"] == 1


def test_regression_cases_accept_missing_predictions():
    parent = [
        {
            "sample_id": "a",
            "gold": {"responsible_agent": "agent", "decisive_step": 1},
            "prediction": {"responsible_agent": "agent", "evidence_step_ids": [1]},
            "turns": 1,
            "inspected_step_ids": [1],
            "error": "",
        }
    ]
    child = [
        {
            "sample_id": "a",
            "gold": {"responsible_agent": "agent", "decisive_step": 1},
            "prediction": None,
            "turns": 1,
            "inspected_step_ids": [],
            "error": "max turns reached",
        }
    ]
    assert regression_cases(parent, child)[0]["child_prediction"] is None


def test_selection_shortlists_without_changing_active_harness(tmp_path, monkeypatch):
    generated_root = tmp_path / "generated"
    candidate_name = "candidate_selection_test"
    candidate = generated_root / candidate_name
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "harnesses/who_when/initial_harness",
        candidate,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    monkeypatch.setattr(
        generated_harnesses,
        "__path__",
        [*generated_harnesses.__path__, str(generated_root)],
    )

    parent_records = [
        {
            "sample_id": "a",
            "gold": {"responsible_agent": "agent_a", "decisive_step": 1},
            "prediction": {"responsible_agent": "agent_a", "evidence_step_ids": [1]},
            "turns": 2,
            "inspected_step_ids": [1],
            "error": "",
        },
        {
            "sample_id": "b",
            "gold": {"responsible_agent": "agent_b", "decisive_step": 2},
            "prediction": {"responsible_agent": "wrong", "evidence_step_ids": [1]},
            "turns": 2,
            "inspected_step_ids": [1],
            "error": "",
        },
    ]
    child_records = [
        {
            **record,
            "prediction": {
                "responsible_agent": record["gold"]["responsible_agent"],
                "evidence_step_ids": [record["gold"]["decisive_step"]],
            },
        }
        for record in parent_records
    ]
    parent_path = tmp_path / "parent.jsonl"
    child_path = tmp_path / "child.jsonl"
    parent_path.write_text(
        "".join(json.dumps(record) + "\n" for record in parent_records),
        encoding="utf-8",
    )
    child_path.write_text(
        "".join(json.dumps(record) + "\n" for record in child_records),
        encoding="utf-8",
    )
    config_path = tmp_path / "base.yaml"
    config_path.write_text("harness:\n  active: initial_harness\n", encoding="utf-8")
    registry_path = tmp_path / "tree.json"
    smoke = smoke_test_harness(candidate_name)
    assert smoke.passed
    register_candidate(
        registry_path,
        candidate_name,
        "initial_harness",
        candidate,
        smoke,
    )

    result = select_candidate(
        config_path=config_path,
        registry_path=registry_path,
        parent_name="initial_harness",
        child_name=candidate_name,
        stages=[(parent_path, child_path)],
        report_path=tmp_path / "selection.json",
        judge=StrictAttributionJudge(),
    )

    assert result["decision"] == "shortlist_child"
    assert result["coverage_is_hard_gate"] is False
    assert active_harness(load_config(config_path)) == "initial_harness"
    tree = load_harness_tree(registry_path)
    assert tree["active"] == "initial_harness"
    assert tree["nodes"][candidate_name]["selections"][0]["report"] == "selection.json"


def test_meta_workspace_persists_reused_failure_feedback_before_backend(tmp_path, monkeypatch):
    import pytest

    from src.meta_evolver import runner

    source = tmp_path / "harnesses/initial_harness"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("def run():\n    return 1\n")
    traces = tmp_path / "rollouts.jsonl"
    traces.write_text('{"sample_id":"case"}\n')
    feedback = {"failure_causes": ["regression"], "evidence": {"case": "case"}}
    config = {
        "project": {"domain": "workspace_test"},
        "harness": {
            "kind": "workspace_test",
            "package": "harnesses.workspace_test",
            "root": "harnesses",
            "base": "initial_harness",
            "active": "initial_harness",
        },
        "meta_evolver": {"backend": "codex", "codex": {"command": "unused"}},
    }
    monkeypatch.setattr(runner, "_require_command", lambda command: None)
    monkeypatch.setattr(runner, "_latest_failure_feedback", lambda *args: feedback)
    monkeypatch.setattr(
        runner,
        "get_domain_plugin",
        lambda config: SimpleNamespace(
            build_evidence=lambda source, output: output.write_text(source.read_text()),
            meta_user_prompt=lambda config: "test domain",
        ),
    )

    def stop_before_backend(root, reference, parent):
        assert (
            json.loads((reference.parent / "prior_failure_feedback.json").read_text()) == feedback
        )
        raise RuntimeError("test stopped before backend")

    monkeypatch.setattr(runner, "_prepare_reference_files", stop_before_backend)
    with pytest.raises(RuntimeError, match="test stopped before backend"):
        runner.run_meta_evolver(config, tmp_path, traces)
