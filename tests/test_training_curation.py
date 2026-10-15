import json
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.domains.base import BaseDomainPlugin
from src.training import curation
from src.utils.persistence import atomic_json, atomic_jsonl


def record(sample_id, *, attempt=1):
    return {
        "sample_id": sample_id,
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "reason",
                "tool_calls": [
                    {
                        "id": "call",
                        "type": "function",
                        "function": {"name": "final_answer", "arguments": "{}"},
                    }
                ],
            }
        ],
        "tools": [{"type": "function", "function": {"name": "final_answer"}}],
        "sampling": {"attempt": attempt, "quality_tier": "strict"},
    }


class Plugin(BaseDomainPlugin):
    name = "test"

    def __init__(self):
        self.fallback_calls = 0

    def load_data(self, path):
        return list(map(json.loads, path.read_text().splitlines()))

    def curation_profile(self, sample):
        return {
            "sample_id": sample["sample_id"],
            "question_text": sample["question"],
            "question": {"task": sample["question"]},
            "hard": sample.get("hard", False),
            "features": {"difficulty": "hard" if sample.get("hard") else "easy"},
            "group_keys": sample.get("group_keys", []),
        }

    def prepare_training_source(self, request):
        self.fallback_calls += 1
        assert request.sampling_config["balance_reference_data"].endswith("val.jsonl")
        # Exercise restoration of hard records dropped by the old heuristic.
        output = request.output_dir / "balanced.jsonl"
        atomic_jsonl(output, [record("easy")])
        return output


@pytest.fixture
def inputs(tmp_path):
    train, val, accepted = [
        tmp_path / name for name in ("train.jsonl", "val.jsonl", "accepted.jsonl")
    ]
    atomic_jsonl(
        train,
        [
            {"sample_id": "hard", "question": "Difficult training task", "hard": True},
            {"sample_id": "easy", "question": "Easy training task"},
        ],
    )
    atomic_jsonl(val, [{"sample_id": "val", "question": "Validation task"}])
    atomic_jsonl(accepted, [record("hard"), record("easy")])
    return {
        "plugin": Plugin(),
        "accepted_path": accepted,
        "train_data": train,
        "val_data": val,
        "training_dataset": tmp_path / "sft.jsonl",
        "harness_name": "h",
        "quality_tier": "strict",
        "sampling_config": {},
        "training_config": {
            "data_curation": {"mode": "agent", "min_records": 5, "max_attempts": 2}
        },
        "config": {"meta_evolver": {}},
        "project_root": tmp_path,
        "validation_python": sys.executable,
    }


def fake_agent(monkeypatch, plans):
    calls = []

    def run(**kwargs):
        calls.append(deepcopy(kwargs["inputs"]))
        questions = kwargs["input_files"]["val_questions.jsonl"].read_text()
        assert "Validation task" in questions
        plan = plans[len(calls) - 1]
        if isinstance(plan, Exception):
            raise plan
        workspace = kwargs["output_dir"] / "workspace"
        atomic_json(workspace / "plan.json", plan)
        result = workspace / "final_message.md"
        atomic_json(result, {"plan_file": "plan.json", "rationale": "mixture"})
        return result

    monkeypatch.setattr("src.training.curation_agent.run_curation_agent", run)
    return calls


def plan(count=4):
    return {"default_count": 1, "counts": {"r000000": count}, "rationale": "repeat hard case"}


def test_agent_receives_validation_feedback_then_real_arrow_gate_passes(inputs, monkeypatch):
    pytest.importorskip("datasets")
    original = inputs["accepted_path"].read_bytes()
    calls = fake_agent(monkeypatch, [plan(0), plan()])
    count, source = curation.curate_dataset(**inputs)
    assert count == 5 and len(calls) == 2
    assert "protected hard record" in calls[1]["previous_errors"][0]["error"]
    rows = list(map(json.loads, source.read_text().splitlines()))
    assert [row["sample_id"] for row in rows].count("hard") == 4
    assert rows[0]["messages"] == record("hard")["messages"]
    assert inputs["accepted_path"].read_bytes() == original
    report = json.loads((source.parent / "report.json").read_text())
    assert report["unique_samples"] == 2 and report["repeated_records"] == 3
    assert report["mode"] == "agent"
    # The same immutable receipt resumes without another model request.
    assert curation.curate_dataset(**inputs) == (count, source)
    assert len(calls) == 2


def test_fallback_restores_hard_examples_and_repeats_to_1000(inputs, monkeypatch):
    inputs["training_config"]["data_curation"]["min_records"] = 1000
    fake_agent(monkeypatch, [RuntimeError("provider unavailable")] * 2)
    checked = []
    monkeypatch.setattr(
        curation, "validate_arrow", lambda path, count, python: checked.append(count)
    )
    count, source = curation.curate_dataset(**inputs)
    assert count == 1000 and checked == [1000]
    rows = list(map(json.loads, source.read_text().splitlines()))
    assert sum(row["sample_id"] == "hard" for row in rows) == 999
    assert inputs["plugin"].fallback_calls == 1
    report = json.loads((source.parent / "report.json").read_text())
    assert report["mode"] == "heuristic_fallback" and report["unique_samples"] == 2


def test_arrow_failure_returns_feedback_and_cannot_publish_even_fallback(inputs, monkeypatch):
    calls = fake_agent(monkeypatch, [plan(), plan()])

    def reject(*args):
        raise ValueError("Arrow column mismatch")

    monkeypatch.setattr(curation, "validate_arrow", reject)
    inputs["training_dataset"].write_text("previous dataset\n")
    with pytest.raises(RuntimeError, match="SFT blocked"):
        curation.curate_dataset(**inputs)
    assert "Arrow column mismatch" in calls[1]["previous_errors"][0]["error"]
    assert inputs["training_dataset"].read_text() == "previous dataset\n"


def test_overlap_excludes_renamed_questions_groups_and_unknown_samples(inputs):
    rows = list(map(json.loads, inputs["train_data"].read_text().splitlines()))
    rows += [
        {"sample_id": "renamed", "question": " VALIDATION, task!! "},
        {"sample_id": "group", "question": "Different phrasing", "group_keys": ["g"]},
    ]
    atomic_jsonl(inputs["train_data"], rows)
    atomic_jsonl(
        inputs["val_data"],
        [{"sample_id": "val", "question": "Validation task", "group_keys": ["g"]}],
    )
    atomic_jsonl(
        inputs["accepted_path"],
        [record("hard"), record("hard"), record("renamed"), record("group"), record("outside")],
    )
    pool, _, excluded = curation.build_inventory(
        inputs["plugin"], inputs["accepted_path"], inputs["train_data"], inputs["val_data"]
    )
    assert len(pool) == 1 and len(excluded) == 4
    assert sum("overlap" in row["error"] for row in excluded) == 2


@pytest.mark.parametrize(
    "change,match",
    [
        ({"r000001": 4}, "only hard"),
        ({"unknown": 1}, "unknown record"),
        ({"r000000": True}, "integer"),
        ({"r000000": 0}, "protected hard"),
    ],
)
def test_plan_rejects_invalid_counts(inputs, change, match):
    pool, _, _ = curation.build_inventory(
        inputs["plugin"], inputs["accepted_path"], inputs["train_data"], inputs["val_data"]
    )
    proposal = plan()
    proposal["counts"].update(change)
    with pytest.raises(ValueError, match=match):
        curation.validate_plan(proposal, pool, minimum=5, maximum=20)


def test_raw_whowhen_answer_is_checked_before_harness_reordering():
    from src.domains.who_when.curation import validate_record
    from src.traces import AgentTrace

    sample = AgentTrace("s", "task", [], True, decisive_step=6, responsible_agents=["A"])
    row = record("s")
    row["sampling"]["inspected_step_ids"] = [6, 3]
    function = row["messages"][0]["tool_calls"][0]["function"]
    function["arguments"] = json.dumps(
        {"is_failure": True, "evidence_step_ids": [7, 6, 3], "responsible_agent": "A"}
    )
    with pytest.raises(ValueError, match="raw final_answer"):
        validate_record(row, sample)
    function["arguments"] = json.dumps(
        {"is_failure": True, "evidence_step_ids": [6, 7, 3], "responsible_agent": "A"}
    )
    validate_record(row, sample)


def test_meta_text_stages_question_files_and_isolates_process(tmp_path, monkeypatch):
    from src.meta_evolver import context

    question = tmp_path / "question.jsonl"
    question.write_text('{"question":"visible validation problem"}\n')
    output = tmp_path / "session"
    monkeypatch.setattr(context, "_require_command", lambda command: None)

    def run(command, **kwargs):
        assert "isolation.py" in command[1]
        assert (output / "workspace/questions.jsonl").read_text() == question.read_text()
        atomic_json(
            output / "workspace/final_message.md", {"plan_file": "plan.json", "rationale": "ok"}
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(context.subprocess, "run", run)
    config = {"meta_evolver": {"backend": "codex", "codex": {"command": "fake"}}}
    context.run_meta_text(
        config=config,
        project_root=tmp_path,
        output_dir=output,
        inputs={},
        prompt="curate",
        input_files={"questions.jsonl": question},
        isolate=True,
        output_schema=curation.RESULT_SCHEMA,
    )
    question.write_text("changed\n")
    with pytest.raises(ValueError, match="staged MetaAgent input changed"):
        context.run_meta_text(
            config=config,
            project_root=tmp_path,
            output_dir=output,
            inputs={},
            prompt="curate",
            input_files={"questions.jsonl": question},
        )


def test_minimum_cannot_be_padded_with_easy_examples(inputs, monkeypatch):
    atomic_jsonl(inputs["train_data"], [{"sample_id": "easy", "question": "Easy training task"}])
    atomic_jsonl(inputs["accepted_path"], [record("easy")])
    fake_agent(monkeypatch, [RuntimeError("unavailable")] * 2)
    with pytest.raises(RuntimeError, match="SFT blocked"):
        curation.curate_dataset(**inputs)
    assert not inputs["training_dataset"].exists()


def test_round_training_gate_preserves_sampling_and_never_launches_sft(tmp_path):
    from src.meta_evolver.round_runtime import Runtime

    runtime = Runtime.__new__(Runtime)
    runtime.round_id = "round_1"
    runtime.round_dir = tmp_path / "round_1"
    runtime.root = tmp_path
    runtime.train_data = tmp_path / "train.jsonl"
    runtime.val_data = tmp_path / "val.jsonl"
    runtime.spec = SimpleNamespace(domain="test")
    runtime.config = {"rejection_sampling": {"quality_tier": "strict"}}
    runtime.training_config = {"data_curation": {"mode": "agent"}}
    runtime.cost = lambda *args, **kwargs: None
    called = []

    def prepare(**kwargs):
        called.append(kwargs)
        raise RuntimeError("data gate blocked SFT")

    def launch(*args, **kwargs):
        pytest.fail("must not launch SFT after dataset gate failure")

    runtime.services = SimpleNamespace(
        prepare_training_dataset=prepare, run=launch, training_python=sys.executable
    )
    state_path = runtime.round_dir / "training/candidate_1/training_state.json"
    atomic_json(state_path, {"sampling_pair_id": "p", "sampling_complete": True})
    state = runtime.train({"id": "p", "candidate_rank": 1, "harness": "h"})
    assert state["sampling_complete"] and state["error"] == "data gate blocked SFT"
    assert not state.get("dataset_complete") and not state.get("sft_started")
    assert called[0]["val_data"] == runtime.val_data
    assert called[0]["config"] is runtime.config
