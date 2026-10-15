import json
import threading
import time

from src.domains.round_adapter import EvaluationArtifact, get_round_adapter
from src.domains.who_when.runtime.infer import (
    _load_resumable_records,
    run_rollout_with_retries,
)
from src.domains.who_when.runtime.rollout import Rollout
from src.runner.execution import ItemProgress, in_completion_order
from src.traces import AgentTrace, DetectionResult


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _prediction(sample_id):
    return {
        "sample_id": sample_id,
        "gold": {"is_failure": True, "decisive_step": 1, "responsible_agent": "agent"},
        "prediction": {
            "is_failure": True,
            "evidence_step_ids": [1],
            "responsible_agent": "agent",
        },
        "error": "",
    }


def test_whowhen_adapter_requires_complete_rollouts_and_matching_metrics(tmp_path):
    artifact = EvaluationArtifact(
        tmp_path,
        tmp_path / "output.jsonl",
        tmp_path / "metrics.json",
    )
    adapter = get_round_adapter("who_when")
    _write_jsonl(artifact.rollouts, [_prediction("a")])
    artifact.metrics.write_text(
        json.dumps(
            {
                "protocol": "who_and_when_official",
                "total": 2,
                "agent_accuracy": 1.0,
                "step_accuracy": 1.0,
                "coverage": 1.0,
            }
        ),
        encoding="utf-8",
    )
    assert not adapter.artifact_complete(artifact, expected=2)

    _write_jsonl(artifact.rollouts, [_prediction("b"), _prediction("a")])
    assert adapter.artifact_complete(artifact, expected=2)
    assert not adapter.artifact_complete(artifact, expected=3)

    artifact.rollouts.write_text(
        artifact.rollouts.read_text(encoding="utf-8") + "{",
        encoding="utf-8",
    )
    assert not adapter.artifact_complete(artifact, expected=2)


def test_resume_reuses_only_complete_attributions(tmp_path):
    output = tmp_path / "output.jsonl"
    _write_jsonl(
        output,
        [
            _prediction("complete"),
            {"sample_id": "failed", "prediction": None, "error": "HTTP 400"},
        ],
    )
    output.write_text(output.read_text(encoding="utf-8") + "{", encoding="utf-8")
    assert set(_load_resumable_records(output)) == {"complete"}


def test_rollout_retries_until_attribution_is_complete(monkeypatch):
    trace = AgentTrace(
        sample_id="sample",
        task="task",
        steps=[],
        is_failure=True,
        decisive_step=1,
        responsible_agents=["agent"],
    )
    calls = []

    def fake_run_rollout(*_args, **_kwargs):
        calls.append(None)
        result = None
        error = "assistant returned no tool call"
        if len(calls) == 3:
            result = DetectionResult(
                is_failure=True,
                evidence_step_ids=(1,),
                responsible_agent="agent",
            )
            error = ""
        return Rollout("sample", [], result, [], 1, error)

    monkeypatch.setattr("src.domains.who_when.runtime.infer.run_rollout", fake_run_rollout)
    rollout = run_rollout_with_retries(
        trace,
        object(),
        max_turns=2,
        harness_name="initial_harness",
        attempts=3,
    )
    assert len(calls) == 3
    assert rollout.result is not None


def test_inference_work_is_concurrent():
    barrier = threading.Barrier(2)

    def worker(value):
        barrier.wait(timeout=1)
        time.sleep(0.01)
        return value

    assert sorted(in_completion_order([1, 2], worker, concurrency=2)) == [1, 2]


def test_shared_item_progress_has_stable_detailed_fields():
    messages = []

    class Logger:
        def info(self, _format, message):
            messages.append(message)

    progress = ItemProgress(Logger(), total=2)
    progress.log(sample_id="a", score=True, reused=False, error=False)

    assert messages == [
        "item=1/2 sample_id=a score=1.000000 reused=False error=False"
    ]
