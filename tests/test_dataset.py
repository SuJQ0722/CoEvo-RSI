import json

from src.domains.who_when.runtime.build_dataset import (
    attribution_validation_eligible,
    build_splits,
    choose_validation_samples,
    evaluation_overlap,
    normalize_who_and_when,
    overlaps_evaluation,
    task_identity,
)


def test_same_benchmark_task_has_same_identity():
    first = {
        "task_dataset": "benchmark",
        "task_id": "task-1",
        "source_dataset": "source-a",
        "source_id": "trajectory-1",
    }
    second = {
        **first,
        "source_dataset": "source-b",
        "source_id": "trajectory-2",
    }
    assert task_identity(first) == task_identity(second)


def test_validation_selection_is_random_reproducible_and_exact_size():
    records = [{"sample_id": value} for value in "abc"]
    selected = choose_validation_samples(records, 2, seed=7)
    assert len(selected) == 2
    assert selected == choose_validation_samples(records, 2, seed=7)


def test_attribution_validation_requires_failure_agent_and_step():
    assert attribution_validation_eligible(
        {"is_failure": True, "responsible_agent": "agent", "decisive_step": 0}
    )
    assert not attribution_validation_eligible(
        {"is_failure": False, "responsible_agent": "agent", "decisive_step": 0}
    )
    assert not attribution_validation_eligible(
        {"is_failure": True, "responsible_agent": "", "decisive_step": 0}
    )


def test_who_and_when_steps_are_normalized_to_one_based_ids(tmp_path):
    source = tmp_path / "Algorithm-Generated.jsonl"
    source.write_text(
        json.dumps(
            {
                "question_ID": "q1",
                "question": "question",
                "history": [{"role": "assistant", "content": "bad"}],
                "mistake_agent": "assistant",
                "mistake_step": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record = normalize_who_and_when([source])[0]
    assert record["trajectory"][0]["step_id"] == 1
    assert record["decisive_step"] == 1


def test_who_and_when_question_overlap_is_detected():
    row = {"task": "Instructions\nThe exact evaluation question here.\nAnswer it."}
    assert overlaps_evaluation(row, {"the exact evaluation question here."})
    assert (
        evaluation_overlap(row, {"the exact evaluation question here."})["match_type"]
        == "wrapped_contains_question"
    )


def test_who_and_when_exact_task_is_distinguished_from_wrapped_prompt():
    overlap = evaluation_overlap(
        {"task": "The exact evaluation question here."},
        {"the exact evaluation question here."},
    )
    assert overlap is not None
    assert overlap["match_type"] == "exact_task"


def test_build_splits_writes_who_and_when_overlap_audit(tmp_path):
    source = tmp_path / "source.jsonl"
    who_and_when = tmp_path / "who.jsonl"
    output_dir = tmp_path / "output"
    source_rows = [
        {
            "sample_id": "same-task-different-trajectory",
            "source_dataset": "source-a",
            "source_id": "1",
            "task_dataset": "benchmark-a",
            "task_id": "task-1",
            "task": "The exact evaluation question here.",
            "trajectory": [{"step_id": 0, "content": "different trajectory"}],
            "responsible_agent": "agent-a",
            "decisive_step": 0,
            "failure_reason": "same reason",
            "is_failure": True,
        },
        {
            "sample_id": "leaked",
            "source_dataset": "source-a",
            "source_id": "2",
            "task_dataset": "benchmark-a",
            "task_id": "task-1-copy",
            "task": "A differently wrapped task is allowed here.",
            "trajectory": [
                {
                    "step_id": 0,
                    "role": "assistant",
                    "name": "agent-a",
                    "content": "gold trajectory",
                }
            ],
            "responsible_agent": "agent-a",
            "decisive_step": 0,
            "failure_reason": "same reason",
            "is_failure": True,
        },
        {
            "sample_id": "clean",
            "source_dataset": "source-b",
            "source_id": "3",
            "task_dataset": "benchmark-b",
            "task_id": "task-2",
            "task": "An unrelated training task.",
            "trajectory": [{"step_id": 0, "content": "clean"}],
            "is_failure": True,
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8")
    who_and_when.write_text(
        json.dumps(
            {
                "question_ID": "q1",
                "question": "The exact evaluation question here.",
                "history": [
                    {
                        "role": "assistant",
                        "name": "agent-a",
                        "content": "gold trajectory",
                    }
                ],
                "mistake_agent": "agent-a",
                "mistake_step": 0,
                "mistake_reason": "same reason",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    clean_output = tmp_path / "merged_clean.jsonl"
    counts = build_splits([source], output_dir, 0, [who_and_when], clean_output=clean_output)
    audit = [
        json.loads(line)
        for line in (output_dir / "excluded_who_and_when_overlap.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert counts["excluded_who_and_when_overlap"] == 1
    assert counts["who_and_when_task_overlap"] == 1
    assert counts["train"] == 2
    assert counts["auxiliary_detection_training_pool"] == 1
    assert counts["val"] == 0
    assert counts["test"] == 1
    assert audit[0]["sample_id"] == "leaked"
    assert audit[0]["match_type"] == "exact_trajectory_and_annotation"
    assert audit[0]["matched_who_and_when_references"] == ["who:q1"]
    assert len(clean_output.read_text(encoding="utf-8").splitlines()) == 2
