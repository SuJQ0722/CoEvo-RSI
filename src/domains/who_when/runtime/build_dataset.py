"""Build leakage-safe train/validation splits and a Who&When test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.utils.persistence import atomic_json, atomic_jsonl


def iter_records(paths: Iterable[Path]):
    seen: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = str(record.get("sample_id", ""))
                if not sample_id:
                    raise ValueError(f"{path}:{line_number}: missing sample_id")
                if sample_id in seen:
                    raise ValueError(f"duplicate sample_id: {sample_id}")
                if not isinstance(record.get("is_failure"), bool):
                    raise ValueError(f"{path}:{line_number}: is_failure must be boolean")
                if not record.get("trajectory"):
                    raise ValueError(f"{path}:{line_number}: trajectory is empty")
                seen.add(sample_id)
                yield record


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def task_identity(record: dict) -> str:
    if record.get("task_group_id"):
        return str(record["task_group_id"])
    if record.get("task_dataset") and record.get("task_id"):
        value = ("task", record["task_dataset"], record["task_id"])
    else:
        value = ("source", record.get("source_dataset"), record.get("source_id"))
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def iter_who_and_when(paths: Iterable[Path]):
    for path in paths:
        if path.is_dir():
            for item in sorted(path.glob("*.json"), key=lambda value: int(value.stem)):
                yield path.name, json.loads(item.read_text(encoding="utf-8"))
        else:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield path.stem, json.loads(line)


def load_evaluation_questions(paths: Iterable[Path]) -> set[str]:
    return set(load_evaluation_question_index(paths))


def load_evaluation_question_index(paths: Iterable[Path]) -> dict[str, list[str]]:
    questions: dict[str, list[str]] = defaultdict(list)
    for split, row in iter_who_and_when(paths):
        question = normalize_text(row.get("question", ""))
        if question:
            source_id = str(row.get("question_ID") or row.get("id") or "unknown")
            reference = f"{split}:{source_id}"
            if reference not in questions[question]:
                questions[question].append(reference)
    return dict(questions)


def evaluation_overlap(record: dict, questions: set[str]) -> dict | None:
    task = normalize_text(record.get("task", ""))
    matches = sorted(question for question in questions if question and question in task)
    if not matches:
        return None
    return {
        "match_type": "exact_task" if task in questions else "wrapped_contains_question",
        "matched_questions": matches,
    }


def overlaps_evaluation(record: dict, questions: set[str]) -> bool:
    return evaluation_overlap(record, questions) is not None


def stable_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def canonical_trajectory(trajectory: object) -> list[object]:
    """Remove only synthetic step indices before strict trajectory comparison."""
    canonical: list[object] = []
    for step in trajectory if isinstance(trajectory, list) else []:
        if isinstance(step, dict):
            canonical.append(
                {
                    key: value
                    for key, value in step.items()
                    if key not in {"step_id", "step", "id", "index"}
                }
            )
        else:
            canonical.append(step)
    return canonical


def canonical_decisive_step(value: object) -> int | str | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value).strip()


def canonical_annotation(
    responsible_agent: object,
    decisive_step: object,
    failure_reason: object,
) -> dict[str, object]:
    return {
        "responsible_agent": None
        if responsible_agent is None
        else re.sub(r"\s+", " ", str(responsible_agent)).strip(),
        "decisive_step": canonical_decisive_step(decisive_step),
        "failure_reason": None
        if failure_reason is None
        else re.sub(r"\s+", " ", str(failure_reason)).strip(),
    }


def trajectory_annotation_signature(trajectory: object, annotation: dict) -> str:
    return stable_hash(
        {
            "trajectory": canonical_trajectory(trajectory),
            "annotation": annotation,
        }
    )


def who_and_when_signature_index(paths: Iterable[Path]) -> dict[str, list[str]]:
    signatures: dict[str, list[str]] = defaultdict(list)
    for split, raw in iter_who_and_when(paths):
        source_id = str(raw.get("question_ID") or raw.get("id") or "unknown")
        signature = trajectory_annotation_signature(
            raw.get("history") or [],
            canonical_annotation(
                raw.get("mistake_agent"),
                raw.get("mistake_step"),
                raw.get("mistake_reason"),
            ),
        )
        signatures[signature].append(f"{split}:{source_id}")
    return dict(signatures)


def normalized_who_trajectory(raw: dict) -> list[dict]:
    trajectory = []
    # Who&When stores mistake_step as a zero-based history position. Normalize
    # both the trajectory and the gold label to CoEvoX's one-based step IDs.
    for step_id, step in enumerate(raw.get("history") or [], start=1):
        if isinstance(step, dict):
            trajectory.append({**step, "step_id": step_id})
    return trajectory


def choose_validation_samples(records: list[dict], target: int, seed: int = 42) -> set[str]:
    """Randomly select a reproducible number of trajectory samples."""
    if target < 0 or target > len(records):
        raise ValueError(
            f"validation sample count must be between 0 and {len(records)}; got {target}"
        )
    sample_ids = sorted(record["sample_id"] for record in records)
    return set(random.Random(seed).sample(sample_ids, target))


def attribution_validation_eligible(record: dict) -> bool:
    """Require the same attribution labels used by the Who&When test task."""

    return bool(
        record.get("is_failure") is True
        and str(record.get("responsible_agent") or "").strip()
        and record.get("decisive_step") not in (None, "")
    )


def normalize_who_and_when(paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for index, (split, raw) in enumerate(iter_who_and_when(paths), start=1):
        source_id = str(raw.get("question_ID") or index)
        trajectory = normalized_who_trajectory(raw)
        try:
            decisive_step = int(raw.get("mistake_step")) + 1
        except (TypeError, ValueError):
            decisive_step = None
        rows.append(
            {
                "schema_version": "coevox.failure.v1",
                "sample_id": f"who_and_when:{split}:{source_id}",
                "source_dataset": "Who&When",
                "source_id": source_id,
                "task_id": source_id,
                "task_dataset": "Who&When",
                "benchmark": "Who&When",
                "task": raw.get("question", ""),
                "trajectory": trajectory,
                "tool_logs": [],
                "responsible_agent": raw.get("mistake_agent"),
                "decisive_step": decisive_step,
                "all_error_steps": [decisive_step] if decisive_step is not None else [],
                "failure_reason": raw.get("mistake_reason"),
                "annotation_source": f"Who&When {split}",
                "natural_or_injected": "natural",
                "is_failure": True,
                "is_safe": False,
                "label_scope": "native_who_and_when",
                "metadata": {"who_and_when_split": split},
            }
        )
    return rows


def attribution_signature(record: dict) -> str:
    return stable_hash(
        {
            "is_failure": record.get("is_failure"),
            "is_safe": record.get("is_safe"),
            "responsible_agent": record.get("responsible_agent"),
            "decisive_step": record.get("decisive_step"),
            "all_error_steps": record.get("all_error_steps") or [],
            "failure_type": record.get("failure_type"),
            "failure_reason": record.get("failure_reason"),
            "supporting_evidence": record.get("supporting_evidence"),
            "ideal_action": record.get("ideal_action"),
        }
    )


def deduplicate_trajectories(
    records: list[dict],
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Deduplicate exact trajectories and drop ambiguous label conflicts."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[stable_hash(record.get("trajectory") or [])].append(record)

    clean: list[dict] = []
    audit: list[dict] = []
    duplicate_groups = conflicting_groups = removed_rows = 0
    for trajectory_hash, group in groups.items():
        if len(group) == 1:
            clean.append(group[0])
            continue
        duplicate_groups += 1
        signatures = {attribution_signature(record) for record in group}
        conflict = len(signatures) > 1
        conflicting_groups += int(conflict)
        ordered = sorted(group, key=lambda record: record["sample_id"])
        kept = None if conflict else ordered[0]
        if kept is not None:
            clean.append(kept)
        removed_rows += len(group) if conflict else len(group) - 1
        audit.append(
            {
                "trajectory_hash": trajectory_hash,
                "group_size": len(group),
                "annotation_conflict": conflict,
                "kept_sample_id": kept["sample_id"] if kept else None,
                "sample_ids": [record["sample_id"] for record in ordered],
            }
        )
    clean.sort(key=lambda record: record["sample_id"])
    return (
        clean,
        audit,
        {
            "input_rows": len(records),
            "output_rows": len(clean),
            "exact_trajectory_duplicate_groups": duplicate_groups,
            "conflicting_annotation_groups": conflicting_groups,
            "removed_rows": removed_rows,
        },
    )


def build_splits(
    inputs: list[Path],
    output_dir: Path,
    validation_count: int,
    who_and_when_paths: list[Path],
    clean_output: Path | None = None,
    validation_seed: int = 42,
) -> dict[str, Any]:
    question_index = load_evaluation_question_index(who_and_when_paths)
    questions = set(question_index)
    who_rows = [raw for _, raw in iter_who_and_when(who_and_when_paths)]
    who_signatures = who_and_when_signature_index(who_and_when_paths)
    leakage_clean: list[dict] = []
    excluded_audit: list[dict] = []
    task_overlap_audit: list[dict] = []
    for record in iter_records(inputs):
        task_overlap = evaluation_overlap(record, questions)
        signature = trajectory_annotation_signature(
            record.get("trajectory") or [],
            canonical_annotation(
                record.get("responsible_agent"),
                record.get("decisive_step"),
                record.get("failure_reason"),
            ),
        )
        matching_references = who_signatures.get(signature, [])
        if task_overlap is not None:
            matched_questions = task_overlap["matched_questions"]
            task_overlap_audit.append(
                {
                    "sample_id": record["sample_id"],
                    "source_dataset": record.get("source_dataset"),
                    "source_id": record.get("source_id"),
                    "task_dataset": record.get("task_dataset"),
                    "task_id": record.get("task_id"),
                    "match_type": task_overlap["match_type"],
                    "matched_questions": matched_questions,
                    "matched_question_references": {
                        question: question_index[question] for question in matched_questions
                    },
                    "exact_trajectory_and_annotation": bool(matching_references),
                }
            )
        if not matching_references:
            leakage_clean.append(record)
            continue
        excluded_audit.append(
            {
                "sample_id": record["sample_id"],
                "source_dataset": record.get("source_dataset"),
                "source_id": record.get("source_id"),
                "task_dataset": record.get("task_dataset"),
                "task_id": record.get("task_id"),
                "match_type": "exact_trajectory_and_annotation",
                "matched_who_and_when_references": matching_references,
            }
        )

    clean, deduplication_audit, deduplication = deduplicate_trajectories(leakage_clean)
    attribution_pool = [record for record in clean if attribution_validation_eligible(record)]
    validation_pool = attribution_pool
    validation_samples = choose_validation_samples(
        validation_pool, validation_count, validation_seed
    )
    validation = [row for row in clean if row["sample_id"] in validation_samples]
    # Validation is attribution-only, but training retains every non-test
    # trajectory. Records without complete who/when labels provide the auxiliary
    # failure-detection objective instead of being discarded.
    train = [row for row in clean if row["sample_id"] not in validation_samples]
    test = normalize_who_and_when(who_and_when_paths)

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output_dir / "deduplication_audit.jsonl", deduplication_audit)
    atomic_jsonl(output_dir / "who_and_when_task_overlap_audit.jsonl", task_overlap_audit)
    atomic_jsonl(output_dir / "excluded_who_and_when_overlap.jsonl", excluded_audit)
    task_overlap_types = Counter(row["match_type"] for row in task_overlap_audit)
    task_overlap_sources = Counter(str(row["source_dataset"]) for row in task_overlap_audit)
    overlap_sources = Counter(str(row["source_dataset"]) for row in excluded_audit)
    clean_source_counts = Counter(str(row.get("source_dataset")) for row in clean)
    clean_task_counts = Counter(str(row.get("task_dataset")) for row in clean)
    counts = {
        "input": len(leakage_clean) + len(excluded_audit),
        "who_and_when_files": len(who_rows),
        "who_and_when_unique_questions": len(questions),
        "who_and_when_task_overlap": len(task_overlap_audit),
        "who_and_when_task_overlap_types": dict(sorted(task_overlap_types.items())),
        "who_and_when_task_overlap_sources": dict(sorted(task_overlap_sources.items())),
        "excluded_who_and_when_overlap": len(excluded_audit),
        "excluded_source_counts": dict(sorted(overlap_sources.items())),
        "deduplication": deduplication,
        "merged_clean": len(clean),
        "attribution_training_and_validation_pool": len(attribution_pool),
        "auxiliary_detection_training_pool": len(clean) - len(attribution_pool),
        "merged_clean_source_counts": dict(sorted(clean_source_counts.items())),
        "merged_clean_task_dataset_counts": dict(sorted(clean_task_counts.items())),
        "validation_seed": validation_seed,
        "validation_samples": len(validation_samples),
        "validation_protocol": "who_and_when_official",
        "validation_attribution_eligible_pool": len(validation_pool),
        "train": atomic_jsonl(output_dir / "train.jsonl", train),
        "val": atomic_jsonl(output_dir / "eval.jsonl", validation),
        "test": atomic_jsonl(output_dir / "test.jsonl", test),
    }
    atomic_json(output_dir / "split.stats.json", counts, sort_keys=True)
    if clean_output is not None:
        clean_output.parent.mkdir(parents=True, exist_ok=True)
        atomic_jsonl(clean_output, clean)
        clean_stats = {
            key: value
            for key, value in counts.items()
            if key
            not in {
                "train",
                "val",
                "test",
                "validation_seed",
                "validation_samples",
            }
        }
        atomic_json(clean_output.with_suffix(".stats.json"), clean_stats, sort_keys=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-count", type=int, default=200)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--clean-output", type=Path)
    parser.add_argument("--who-and-when", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    counts = build_splits(
        args.input,
        args.output_dir,
        args.validation_count,
        args.who_and_when,
        args.clean_output,
        args.validation_seed,
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
