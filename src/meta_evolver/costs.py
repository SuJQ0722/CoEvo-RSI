"""Usage accounting from existing CLI and rollout receipts, without invented prices."""

import json
from collections import defaultdict
from pathlib import Path


def meta_usage(path: Path):
    usages = []
    if not path.exists():
        return None
    for line in path.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("type") in {"turn.completed", "result"} and isinstance(
            record.get("usage"), dict
        ):
            usages.append(record["usage"])
    if not usages:
        return None
    return {
        "input_tokens": sum(int(u.get("input_tokens", u.get("prompt_tokens", 0))) for u in usages),
        "output_tokens": sum(
            int(u.get("output_tokens", u.get("completion_tokens", 0))) for u in usages
        ),
        "cached_input_tokens": sum(int(u.get("cached_input_tokens", 0)) for u in usages),
    }


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def _stage_totals(entries):
    totals = defaultdict(
        lambda: {
            "invocations": 0,
            "summed_wall_seconds": 0.0,
            "gpu_hours": 0.0,
            "new_rollouts": 0,
            "task_count": 0,
            "token_cost": 0.0,
        }
    )
    for entry in entries:
        stage = str(entry.get("stage") or "unknown")
        row = totals[stage]
        row["invocations"] += 1
        row["summed_wall_seconds"] += float(entry.get("wall_seconds") or 0)
        row["gpu_hours"] += float(entry.get("gpu_hours") or 0)
        row["new_rollouts"] += int(entry.get("new_rollouts") or entry.get("rollouts") or 0)
        row["task_count"] += int(entry.get("task_count") or 0)
        row["token_cost"] += float(entry.get("token_cost") or 0)
    return {stage: dict(values) for stage, values in sorted(totals.items())}


def ledger_summary(journal, round_dir, config=None):
    entries = list(journal.data.get("costs", []))
    proposals = [p for b in journal.data["batches"] for p in b.get("proposals", [])]
    for proposal in proposals:
        slot = proposal["slot"]
        entries.append(
            {
                "stage": "editor",
                "candidate": proposal["candidate"],
                "wall_seconds": proposal.get("wall_seconds"),
                "valid": proposal.get("valid"),
                "usage": meta_usage(
                    round_dir / f"harness_attempt_{slot:02d}" / "editor" / "stdout.log"
                ),
                "api_cost": None,
                "gpu_hours": 0,
            }
        )
    rejection = dict((config or {}).get("rejection_sampling") or {})
    training_data = []
    rejection_sampling_rollouts = 0
    for rank, training in journal.data.get("training", {}).items():
        directory = round_dir / "training" / f"candidate_{rank}"
        path = directory / "sampling_attempts.jsonl"
        attempts = []
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                attempts = [json.loads(line) for line in stream if line.strip()]
        rejection_sampling_rollouts += len(attempts)
        stats_path = directory / "sampling_stats.json"
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        items = int(stats.get("items") or 0)
        attempts_per_item = int(
            stats.get("attempts_per_item") or rejection.get("attempts_per_item") or 0
        )
        maximum_attempts = items * attempts_per_item if items and attempts_per_item else None
        accepted = int(stats.get("total_accepted_records", stats.get("accepted", 0)) or 0)
        training_id = training.get("training_id")
        stage_entries = [entry for entry in entries if entry.get("training_id") == training_id]
        training_data.append(
            {
                "candidate_rank": int(rank),
                "training_id": training_id,
                "source_items": items,
                "attempts_per_item": attempts_per_item or None,
                "accepted_per_item": rejection.get("accepted_per_item"),
                "maximum_rollout_attempts": maximum_attempts,
                "actual_rollout_attempts": len(attempts),
                "rollout_budget_utilization": (
                    len(attempts) / maximum_attempts if maximum_attempts else None
                ),
                "accepted_records": accepted,
                "strict_accepted_records": int(stats.get("strict_accepted") or 0),
                "repaired_accepted_records": int(stats.get("repaired_accepted") or 0),
                "sft_records": _line_count(directory / "sft_train.jsonl"),
                "sampling_wall_seconds": sum(
                    float(entry.get("wall_seconds") or 0)
                    for entry in stage_entries
                    if entry.get("stage") == "rejection_sampling"
                ),
                "sft_wall_seconds": sum(
                    float(entry.get("wall_seconds") or 0)
                    for entry in stage_entries
                    if entry.get("stage") == "sft"
                ),
                "training_path_wall_seconds": sum(
                    float(entry.get("wall_seconds") or 0)
                    for entry in stage_entries
                    if entry.get("stage") == "training_path"
                ),
            }
        )
        entries.append(
            {
                "stage": "rejection_sampling_usage",
                "candidate_rank": int(rank),
                "rollouts": len(attempts),
                "token_cost": sum(r["total_tokens"] for r in attempts)
                if attempts
                and all(
                    r.get("token_usage_available") and r.get("total_tokens") is not None
                    for r in attempts
                )
                else None,
            }
        )
    return {
        "entries": entries,
        "stage_totals": _stage_totals(entries),
        "duration_accounting": (
            "summed_wall_seconds is summed within each stage; stages such as model_residency "
            "and training_path overlap their child stages and must not be summed into end-to-end time"
        ),
        "generated_candidates": len(proposals),
        "valid_candidates": sum(bool(p.get("valid")) for p in proposals),
        "failure_attributions": sum("attribution" in b for b in journal.data["batches"]),
        "training_paths": len(journal.data.get("training", {})),
        "new_validation_rollouts": sum(
            e.get("new_rollouts", 0) for e in entries if e.get("split") == "val"
        ),
        "rejection_sampling_rollouts": rejection_sampling_rollouts,
        "training_data": training_data,
        "generated_training_records": sum(row["accepted_records"] for row in training_data),
        "sft_training_records": sum(row["sft_records"] for row in training_data),
        "api_prices_available": False,
    }
