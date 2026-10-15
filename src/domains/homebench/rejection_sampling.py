"""Fresh train-only rollouts: exact official correctness, no gold replacement."""

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from src.config import apply_harness_overrides, harness_spec, load_config
from src.reject_sampling import SamplingProgress, policy_attempts
from src.runner.execution import in_completion_order
from src.training.sampling import (
    SamplingRecordWriter,
    SamplingStats,
    accepted_record,
    read_accepted_jsonl,
)
from src.utils.persistence import atomic_json

from .client import build_client
from .data import load
from .evaluator import judge
from .runtime.execution import run_sample
from .runtime.infer import identity
from .submission import official_prediction


def _instruction_count(task_type):
    match = re.match(r"^multi(\d+)_", str(task_type))
    if not match:
        raise ValueError(f"invalid HomeBench multi-operation task type: {task_type}")
    return int(match.group(1))


def balance_training_rollouts(
    *, accepted_path, data_path, reference_path, output_path, seed=42
):
    """Drop only first-attempt easy records when that improves validation-arity fit."""
    records = read_accepted_jsonl(accepted_path)
    task_arity = {sample.sample_id: _instruction_count(sample.record["task_type"]) for sample in load(data_path)}
    reference_counts = Counter(
        _instruction_count(sample.record["task_type"]) for sample in load(reference_path)
    )
    missing = {row["sample_id"] for row in records} - set(task_arity)
    if missing:
        raise ValueError(f"accepted HomeBench records are absent from training data: {len(missing)}")
    target_total = sum(reference_counts.values())
    target = {key: value / target_total for key, value in reference_counts.items()}
    kept = list(records)
    counts = Counter(task_arity[row["sample_id"]] for row in kept)

    def divergence(current, total):
        return sum(abs(current.get(key, 0) / total - target[key]) for key in target)

    easy = defaultdict(list)
    for row in kept:
        sampling = row.get("sampling") or {}
        if (
            int(sampling.get("attempt", 0)) == 1
            and sampling.get("quality_tier") == "strict"
        ):
            easy[task_arity[row["sample_id"]]].append(row)
    rng = random.Random(seed)
    dropped_ids = set()
    before = divergence(counts, len(kept))
    # Progress from shortest/easiest structural group upward. A record is
    # removed only if doing so strictly improves distance to validation arity.
    for arity in sorted(easy):
        rng.shuffle(easy[arity])
        for row in easy[arity]:
            if len(kept) - len(dropped_ids) <= 1:
                break
            total = len(kept) - len(dropped_ids)
            candidate = counts.copy()
            candidate[arity] -= 1
            if divergence(candidate, total - 1) + 1e-15 < divergence(counts, total):
                dropped_ids.add(row["sample_id"])
                counts = candidate
            else:
                break
    selected = [row for row in kept if row["sample_id"] not in dropped_ids]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic_json(
        output_path.with_suffix(".stats.json"),
        {
            "input_records": len(records),
            "output_records": len(selected),
            "dropped_first_attempt_easy": len(dropped_ids),
            "drop_order": sorted(easy),
            "input_instruction_counts": dict(
                sorted(Counter(task_arity[row["sample_id"]] for row in records).items())
            ),
            "output_instruction_counts": dict(sorted(counts.items())),
            "reference_instruction_counts": dict(sorted(reference_counts.items())),
            "l1_before": before,
            "l1_after": divergence(counts, len(selected)),
            "seed": seed,
        },
    )
    return output_path


def collect(request, *, client_factory=None):
    samples = list(load(request.data_path, limit=request.limit))
    if not samples or any(
        s.record["metadata"]["source_split"] != "train"
        or s.record["metadata"].get("label_conflict")
        for s in samples
    ):
        raise ValueError("HomeBench SFT requires clean train samples; validation/test are excluded")
    if request.quality_tier != "strict":
        raise ValueError("HomeBench supports strict rejection sampling only")
    if (
        min(
            request.attempts_per_item,
            request.accepted_per_item,
            request.concurrency,
            request.max_turns,
        )
        < 1
    ):
        raise ValueError("sampling budgets must be positive")
    config = load_config(request.config_path)
    config = apply_harness_overrides(config, harness_spec(config).path_for(request.harness_name))
    if request.temperature is not None:
        config.setdefault("rejection_sampling", {})["temperature"] = request.temperature
    manifest = identity(
        samples,
        config,
        request.harness_name,
        model=request.policy_model,
        base_url=request.base_url,
        max_turns=request.max_turns,
    )
    manifest["sampling"] = {
        "attempts": request.attempts_per_item,
        "accepted": request.accepted_per_item,
        "temperature": config.get("rejection_sampling", {}).get("temperature", 0.7),
    }
    root = request.output_dir
    existing = []
    if request.resume and (root / "accepted_rollouts.jsonl").exists():
        if (
            not (root / "run_manifest.json").exists()
            or json.loads((root / "run_manifest.json").read_text()) != manifest
        ):
            raise ValueError("sampling resume identity mismatch")
        existing = read_accepted_jsonl(root / "accepted_rollouts.jsonl")
    counts = {}
    for row in existing:
        if row.get("task_digest") != manifest["sample_digests"].get(row["sample_id"]):
            raise ValueError("sampling resume task mismatch")
        counts[row["sample_id"]] = counts.get(row["sample_id"], 0) + 1
    atomic_json(root / "run_manifest.json", manifest)
    stats = SamplingStats(
        skipped_existing=sum(
            counts.get(s.sample_id, 0) >= request.accepted_per_item for s in samples
        )
    )
    progress = SamplingProgress(stats, existing_records=len(existing))

    def execute(sample):
        needed = request.accepted_per_item - counts.get(sample.sample_id, 0)

        def run_attempt(attempt):
            client = (
                client_factory(sample)
                if client_factory
                else build_client(
                    config,
                    model=request.policy_model,
                    base_url=request.base_url,
                    sampling=True,
                )
            )
            row = run_sample(
                sample, config, request.harness_name, client, max_turns=request.max_turns
            )
            ok = (
                row["finished"]
                and not row["error"]
                and judge(official_prediction(row), sample.record["target"]["commands"])["passed"]
            )
            reason, record = "wrong_or_incomplete", None
            if row["error"]:
                reason = row["error"]
            if ok:
                try:
                    # Learn the last real request and model response. This also
                    # supports a candidate that prunes or rewrites its context.
                    messages = [*row["requests"][-1]["messages"], row["messages"][-1]]
                    record = accepted_record(
                        sample_id=sample.sample_id,
                        messages=messages,
                        tools=row["requests"][-1]["tools"],
                        sampling={
                            "quality_tier": "strict",
                            "attempt": attempt,
                            "harness": request.harness_name,
                            "model": request.policy_model,
                        },
                        metadata={
                            "domain": "homebench",
                            "task_digest": sample.digest,
                        },
                    )
                    reason = "strict_official_em"
                except (ValueError, KeyError) as exc:
                    reason = f"invalid_training_record:{exc}"
            return record, {
                "sample_id": sample.sample_id,
                "score": int(record is not None),
                "reason": reason,
                "error_detail": row.get("error_detail"),
                "total_tokens": row.get("total_tokens"),
                "token_usage_available": row.get("token_usage_available", False),
            }

        return policy_attempts(
            attempts_per_item=request.attempts_per_item,
            accepted_per_item=needed,
            policy_attempt=run_attempt,
        )

    with SamplingRecordWriter(
        root / "accepted_rollouts.jsonl", root / "sampling_attempts.jsonl", append=request.resume
    ) as writer:
        pending_samples = (
            sample
            for sample in samples
            if counts.get(sample.sample_id, 0) < request.accepted_per_item
        )
        for records, attempts in in_completion_order(
            pending_samples,
            execute,
            request.concurrency,
        ):
            for record in records:
                writer.write_accepted(record)
            for attempt in progress.complete(records, attempts):
                writer.write_attempt(attempt)
                if request.on_attempt:
                    request.on_attempt(attempt)
    atomic_json(
        root / "sampling_stats.json",
        {
            **asdict(stats),
            "existing_records": len(existing),
            "total_accepted_records": len(existing) + stats.accepted,
        },
    )
    return stats


def prepare_training_source(request):
    settings = dict(request.sampling_config or {})
    if not settings.get("balance_by_val_arity", False):
        return request.accepted_path
    reference = Path(str(settings.get("balance_reference_data") or ""))
    if not reference.is_absolute():
        reference = Path.cwd() / reference
    if not reference.is_file():
        raise FileNotFoundError(f"HomeBench balance reference does not exist: {reference}")
    return balance_training_rollouts(
        accepted_path=request.accepted_path,
        data_path=request.data_path,
        reference_path=reference,
        output_path=request.output_dir / "balanced_rollouts.jsonl",
        seed=int(settings.get("balance_seed", 42)),
    )
