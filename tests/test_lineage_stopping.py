"""Round publication counts parent retention, including pre-training failures."""

import json
from types import SimpleNamespace

import pytest

from src.meta_evolver import round_runtime


@pytest.mark.parametrize(
    ("status", "trained", "expected_rollbacks"),
    [
        ("no_update", False, 1),
        ("no_update", True, 1),
        ("untrained_pair", True, 0),
        ("trained_pair", True, 0),
    ],
)
def test_published_round_counts_parent_retention(
    tmp_path, monkeypatch, status, trained, expected_rollbacks
):
    winner = {
        "id": "winner",
        "harness": "parent" if status == "no_update" else "child",
        "model": {"id": "model", "path": str(tmp_path / "model")},
    }
    journal = SimpleNamespace(data={
        "batches": [],
        "training": {"1": {"complete": True}} if trained else {},
        "incumbent_eval": {},
    })
    # Start after archive publication; serving and harness mirrors are unrelated
    # to the summary consumed by the outer early-stopping loop.
    runtime = SimpleNamespace(
        archive_path=tmp_path / "archive.json",
        archive={"rounds": {"round1": {}}, "incumbent": {"pair_id": "winner"}},
        round_id="round1",
        round_dir=tmp_path,
        config_path=tmp_path / "config.yaml",
        registry_path=tmp_path / "registry.json",
        parent="parent",
        parent_source=tmp_path / "parent",
        config={},
        args=SimpleNamespace(round_number=1, stop_after_round=True, base_model_name="base"),
        base_url="http://unused.invalid/v1",
        server=SimpleNamespace(
            process=SimpleNamespace(pid=123), pid_path=tmp_path / "pid", log_path=tmp_path / "log"
        ),
        ensure_model=lambda model: None,
        account_serving=lambda: None,
        inheritance_report=dict,
    )
    monkeypatch.setattr(round_runtime, "set_active_harness", lambda *args: None)
    monkeypatch.setattr(round_runtime, "publish_lineage_harnesses", lambda *args, **kwargs: None)

    summary = round_runtime.Runtime.finalize(runtime, {"status": status, "winner": winner}, journal)

    assert summary["policy_rollbacks"] == expected_rollbacks
    assert summary["policy_updated"] is (status != "no_update")
    assert summary["selected_harness"] == winner["harness"]
    assert json.loads((tmp_path / "round_summary.json").read_text()) == summary
