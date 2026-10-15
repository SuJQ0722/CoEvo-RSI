import json
from pathlib import Path

import pytest
import yaml

from src.runner.full_test import build_plan, discover_evolution, execute_plan


def make_project(tmp_path: Path, *, rounds: dict | None = None) -> tuple[Path, str, str]:
    round_id = "round_03_20260917T120000Z"
    pair_id = "p_test_winner"
    harness = "round_03_candidate"
    model = tmp_path / "model"
    model.mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "registries").mkdir()
    (tmp_path / "harnesses/who_when/initial_harness").mkdir(parents=True)
    (tmp_path / f"harnesses/who_when/candidates/{harness}").mkdir(parents=True)
    data = tmp_path / "data/test.jsonl"
    data.parent.mkdir()
    data.write_text('{"sample_id":"test-1"}\n', encoding="utf-8")
    config = {
        "project": {"domain": "who_when"},
        "model": {"base": str(model)},
        "harness": {
            "kind": "failure_detection",
            "package": "harnesses.who_when",
            "root": "harnesses/who_when",
            "base": "initial_harness",
            "active": "initial_harness",
            "max_turns": 7,
        },
        "data": {"splits": {"test": str(data)}},
        "rollout": {"concurrency": 3},
        "vllm": {
            "command": "vllm",
            "host": "127.0.0.1",
            "port": 8000,
            "max_model_len": 1234,
            "gpu_memory_utilization": 0.8,
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
        },
        "meta_evolver": {"registry_path": "registries/harness_tree.json"},
        "paths": {"output": str(tmp_path / "output/who_when")},
    }
    (tmp_path / "configs/who_when.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    journal = tmp_path / "rounds" / round_id / "journal.json"
    journal.parent.mkdir(parents=True)
    winner = {
        "id": pair_id,
        "harness": harness,
        "model": {"id": "m_test", "path": str(model)},
        "scores": {"agent_accuracy": 50, "step_accuracy": 60},
    }
    journal.write_text(
        json.dumps({"outcome": {"status": "trained_pair", "selected": winner}}),
        encoding="utf-8",
    )
    registry_rounds = rounds or {
        round_id: {
            "status": "trained_pair",
            "journal": str(journal),
            "winner_pair_id": pair_id,
        }
    }
    (tmp_path / "registries/harness_tree.lineage.json").write_text(
        json.dumps({"rounds": registry_rounds}), encoding="utf-8"
    )
    return tmp_path, round_id, pair_id


def test_discovers_round_and_builds_domain_full_test_plan(tmp_path):
    root, round_id, pair_id = make_project(tmp_path)

    selection = discover_evolution(root, pair_id)
    plan = build_plan(root, round_id, orchestrator_python="python")

    assert selection.evolution_id == round_id
    assert selection.pair_id == pair_id
    assert plan.selection.domain == "who_when"
    assert plan.harness == "round_03_candidate"
    assert plan.model_id == "m_test"
    assert plan.max_turns == 7
    assert plan.concurrency == 3
    assert plan.output_dir == root / "output/who_when/full_test" / round_id / plan.harness
    assert "--served-model-name" in plan.vllm_command
    assert plan.vllm_command[plan.vllm_command.index("--served-model-name") + 1] == "m_test"


def test_pair_id_must_identify_one_round(tmp_path):
    root, round_id, pair_id = make_project(tmp_path)
    first = json.loads((root / "registries/harness_tree.lineage.json").read_text())[
        "rounds"
    ][round_id]
    duplicate = dict(first)
    rounds = {round_id: first, "round_04_20260917T120000Z": duplicate}
    (root / "registries/harness_tree.lineage.json").write_text(
        json.dumps({"rounds": rounds}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="ambiguous"):
        discover_evolution(root, pair_id)


def test_dry_run_prints_managed_server_and_domain_commands(tmp_path, capsys):
    root, round_id, _ = make_project(tmp_path)
    plan = build_plan(root, round_id, orchestrator_python="python")

    assert execute_plan(plan, dry_run=True, resume=True) == 0

    output = capsys.readouterr().out
    assert "domain=who_when" in output
    assert "vllm_command=vllm serve" in output
    assert "src.domains.who_when.runtime.infer" in output
    assert "src.domains.who_when.runtime.evaluation" in output
    assert "--resume" in output
    assert not plan.output_dir.exists()


def test_complete_result_is_reused_without_starting_a_server(tmp_path, capsys):
    root, round_id, _ = make_project(tmp_path)
    plan = build_plan(root, round_id, orchestrator_python="python")
    plan.output_dir.mkdir(parents=True)
    plan.artifact.rollouts.write_text(
        '{"sample_id":"test-1"}\n', encoding="utf-8"
    )
    plan.artifact.metrics.write_text(
        json.dumps(
            {
                "protocol": "who_and_when_official",
                "total": 1,
                "agent_accuracy": 75.0,
                "step_accuracy": 80.0,
                "coverage": 1.0,
            }
        ),
        encoding="utf-8",
    )

    assert execute_plan(plan, dry_run=False, resume=True) == 0

    output = capsys.readouterr().out
    assert "resume=complete" in output
    assert '"agent_accuracy": 75.0' in output
    assert (plan.output_dir / "evolution_full_test.json").is_file()
    assert (plan.output_dir / "resolved_config.yaml").is_file()
