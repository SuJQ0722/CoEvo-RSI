from pathlib import Path

from src.config import load_config
from src.domains.round_adapter import get_round_adapter

ROOT = Path(__file__).resolve().parents[1]


def test_round_adapter_dispatches_by_domain():
    who = get_round_adapter(load_config(ROOT / "configs/who_when.yaml"))
    home = get_round_adapter(load_config(ROOT / "configs/homebench.yaml"))
    rbac = get_round_adapter(load_config(ROOT / "configs/rbac.yaml"))

    assert who.metric_names == ("agent_accuracy", "step_accuracy")
    assert who.test_is_derived is False
    assert home.metric_names == ("em", "f1", "precision", "recall")
    assert home.test_is_derived is False
    assert rbac.test_is_derived is False


def test_rbac_adapter_uses_pipeline_python_by_default(tmp_path):
    rbac = get_round_adapter("rbac")

    assert rbac.python_executable({}, tmp_path, "/framework/python") == "/framework/python"


def test_rbac_adapter_allows_explicit_python_environment(tmp_path):
    python = tmp_path / "rbac-env/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    config = {"execution": {"environment_prefix": "rbac-env"}}

    rbac = get_round_adapter("rbac")

    assert rbac.python_executable(config, tmp_path, "/framework/python") == str(python)


def test_whowhen_adapter_owns_focused_validation_and_failure_evaluator(tmp_path):
    adapter = get_round_adapter("who_when")
    commands = adapter.prepare_data_commands(
        python="python",
        project_root=tmp_path,
        paths={},
        splits={"archived_val": "data/archived.jsonl"},
        train_data=tmp_path / "data/train.jsonl",
        val_data=tmp_path / "data/val.jsonl",
        test_data=tmp_path / "data/test.jsonl",
        source_data_override=None,
        who_and_when_root_override=None,
    )
    assert [module for _, module in commands] == ["data.split"]

    artifact = adapter.artifact(tmp_path, "parent_val")
    evaluation = adapter.evaluation_commands(
        python="python",
        config=Path("config.yaml"),
        data=Path("val.jsonl"),
        split="val",
        model="policy",
        base_url="http://localhost:8000/v1",
        harness="candidate",
        max_turns=10,
        concurrency=4,
        artifact=artifact,
        limit=2,
        resume=False,
    )
    assert "src.domains.who_when.runtime.evaluation" in evaluation[1][0]
    assert adapter.metric_names == ("agent_accuracy", "step_accuracy")
