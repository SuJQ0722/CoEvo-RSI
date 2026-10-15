import json
from types import SimpleNamespace

from src.meta_evolver.costs import ledger_summary


def test_ledger_summary_records_training_volume_budget_and_stage_time(tmp_path):
    training = tmp_path / "training" / "candidate_1"
    training.mkdir(parents=True)
    attempts = [
        {"total_tokens": 10, "token_usage_available": True},
        {"total_tokens": 20, "token_usage_available": True},
        {"total_tokens": 30, "token_usage_available": True},
    ]
    (training / "sampling_attempts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in attempts)
    )
    (training / "sft_train.jsonl").write_text("{}\n{}\n")
    (training / "sampling_stats.json").write_text(
        json.dumps(
            {
                "items": 2,
                "attempts": 3,
                "accepted": 2,
                "strict_accepted": 1,
                "repaired_accepted": 1,
                "attempts_per_item": 3,
            }
        )
    )
    journal = SimpleNamespace(
        data={
            "batches": [{"proposals": []}],
            "training": {"1": {"training_id": "train-1"}},
            "costs": [
                {
                    "stage": "rejection_sampling",
                    "training_id": "train-1",
                    "wall_seconds": 12.0,
                },
                {
                    "stage": "sft",
                    "training_id": "train-1",
                    "wall_seconds": 5.0,
                    "gpu_hours": 0.5,
                },
            ],
        }
    )

    result = ledger_summary(
        journal,
        tmp_path,
        {"rejection_sampling": {"attempts_per_item": 3, "accepted_per_item": 1}},
    )

    assert result["rejection_sampling_rollouts"] == 3
    assert result["generated_training_records"] == 2
    assert result["sft_training_records"] == 2
    assert result["training_data"] == [
        {
            "candidate_rank": 1,
            "training_id": "train-1",
            "source_items": 2,
            "attempts_per_item": 3,
            "accepted_per_item": 1,
            "maximum_rollout_attempts": 6,
            "actual_rollout_attempts": 3,
            "rollout_budget_utilization": 0.5,
            "accepted_records": 2,
            "strict_accepted_records": 1,
            "repaired_accepted_records": 1,
            "sft_records": 2,
            "sampling_wall_seconds": 12.0,
            "sft_wall_seconds": 5.0,
            "training_path_wall_seconds": 0,
        }
    ]
    assert result["stage_totals"]["sft"]["summed_wall_seconds"] == 5.0
    assert result["stage_totals"]["sft"]["gpu_hours"] == 0.5
