"""Collect and curate harness rollouts for SFT."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from src.config import active_harness, harness_spec, load_config
from src.domains.who_when.runtime.infer import iter_traces
from src.domains.who_when.runtime.rollout import Rollout, run_rollout
from src.reject_sampling import SamplingProgress, policy_attempts
from src.runner.execution import in_completion_order
from src.runner.model_client import ChatModel, OpenAICompatibleClient
from src.traces import AgentTrace
from src.training.sampling import (
    SamplingStats,
    accepted_record,
    read_accepted_jsonl,
    write_accepted_jsonl,
)
from src.utils.persistence import atomic_json
from harnesses.who_when.initial_harness import tools_for_trace

QualityTier = Literal["strict", "balanced"]

_BALANCE_AXES = ("trace_length", "decisive_position", "agent_count")


def _trace_length_bucket(length: int) -> str:
    if length <= 10:
        return "00_<=10"
    if length <= 20:
        return "01_11-20"
    if length <= 40:
        return "02_21-40"
    if length <= 80:
        return "03_41-80"
    return "04_>80"


def _decisive_position_bucket(trace: AgentTrace) -> str:
    if not trace.steps or trace.decisive_step is None:
        return "04_unavailable"
    step_ids = [step.step_id for step in trace.steps]
    try:
        position = step_ids.index(int(trace.decisive_step)) + 1
    except (TypeError, ValueError):
        return "04_unavailable"
    relative = position / len(step_ids)
    if relative <= 0.25:
        return "00_0-25%"
    if relative <= 0.50:
        return "01_25-50%"
    if relative <= 0.75:
        return "02_50-75%"
    return "03_75-100%"


def _agent_count_bucket(trace: AgentTrace) -> str:
    count = len({step.agent_id for step in trace.steps})
    return str(count) if count < 5 else "5+"


def _difficulty_features(trace: AgentTrace) -> tuple[str, str, str]:
    """Content-free structural proxies available for both train and test."""

    return (
        _trace_length_bucket(len(trace.steps)),
        _decisive_position_bucket(trace),
        _agent_count_bucket(trace),
    )


def _axis_counts(
    features: Iterable[tuple[str, str, str]],
) -> list[Counter[str]]:
    counters = [Counter() for _ in _BALANCE_AXES]
    for values in features:
        for index, value in enumerate(values):
            counters[index][value] += 1
    return counters


def _weighted_tv_distance(
    counts: list[Counter[str]],
    total: int,
    targets: list[dict[str, float]],
    weights: tuple[float, float, float],
) -> float:
    if total <= 0:
        return float("inf")
    weight_total = sum(weights)
    return sum(
        weight
        * 0.5
        * sum(
            abs(counter.get(key, 0) / total - target.get(key, 0.0))
            for key in counter.keys() | target.keys()
        )
        for counter, target, weight in zip(counts, targets, weights, strict=True)
    ) / weight_total


def _serializable_axis_counts(counts: list[Counter[str]]) -> dict[str, dict[str, int]]:
    return {
        axis: dict(sorted(counter.items()))
        for axis, counter in zip(_BALANCE_AXES, counts, strict=True)
    }


def balance_training_rollouts(
    *,
    accepted_path: Path,
    data_path: Path,
    reference_path: Path,
    output_path: Path,
    seed: int = 42,
    max_growth_fraction: float = 0.5,
    max_repeats_per_record: int = 3,
    axis_weights: dict[str, float] | None = None,
) -> Path:
    """Move structural marginals toward validation by repeating verified hard data.

    Every accepted trajectory is retained. Only multi-attempt policy
    trajectories may be repeated, and only while the repeat strictly
    improves the weighted distance to the validation distribution.
    """

    if max_growth_fraction < 0.0:
        raise ValueError("Who&When balance_max_growth_fraction must be non-negative")
    if max_repeats_per_record < 1:
        raise ValueError("Who&When balance_max_repeats_per_record must be positive")
    records = read_accepted_jsonl(accepted_path)
    traces = {trace.sample_id: trace for trace in iter_traces(data_path, None)}
    reference = list(iter_traces(reference_path, None))
    if not records:
        raise ValueError("cannot balance an empty Who&When accepted-rollout file")
    if not reference:
        raise ValueError("Who&When balance reference is empty")
    missing = {str(row.get("sample_id") or "") for row in records} - traces.keys()
    if missing:
        raise ValueError(
            f"accepted Who&When records are absent from training data: {len(missing)}"
        )

    record_features = [
        _difficulty_features(traces[str(record["sample_id"])]) for record in records
    ]
    reference_counts = _axis_counts(_difficulty_features(trace) for trace in reference)
    targets = [
        {key: value / len(reference) for key, value in counter.items()}
        for counter in reference_counts
    ]
    configured_weights = axis_weights or {}
    weights = tuple(
        float(configured_weights.get(axis, 2.0 if axis == "decisive_position" else 1.0))
        for axis in _BALANCE_AXES
    )
    if any(weight <= 0 for weight in weights):
        raise ValueError("Who&When balance axis weights must be positive")

    counts = _axis_counts(record_features)
    before_counts = [counter.copy() for counter in counts]
    before_distance = _weighted_tv_distance(counts, len(records), targets, weights)
    eligible: defaultdict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, (record, features) in enumerate(zip(records, record_features, strict=True)):
        sampling = record.get("sampling") or {}
        if int(sampling.get("attempt", 0)) > 1:
            eligible[features].append(index)

    rng = random.Random(seed)
    for indexes in eligible.values():
        rng.shuffle(indexes)
    eligible_count = sum(len(indexes) for indexes in eligible.values())
    max_additions = int(len(records) * max_growth_fraction)
    repeat_counts: Counter[int] = Counter()
    repeated_indexes: list[int] = []
    distance = before_distance
    while len(repeated_indexes) < max_additions:
        best: tuple[float, tuple[str, str, str], float, list[Counter[str]]] | None = None
        best_record_index: int | None = None
        total_after = len(records) + len(repeated_indexes) + 1
        for features in sorted(eligible):
            available = [
                index
                for index in eligible[features]
                if repeat_counts[index] < max_repeats_per_record
            ]
            if not available:
                continue
            # Spread repeats across distinct hard trajectories before giving a
            # trajectory another copy. The seeded group order breaks ties.
            record_index = min(available, key=lambda index: repeat_counts[index])
            candidate_counts = [counter.copy() for counter in counts]
            for axis_index, value in enumerate(features):
                candidate_counts[axis_index][value] += 1
            candidate_distance = _weighted_tv_distance(
                candidate_counts, total_after, targets, weights
            )
            improvement = distance - candidate_distance
            if improvement <= 1.0e-15:
                continue
            if best is None or improvement > best[0] + 1.0e-15:
                best = (improvement, features, candidate_distance, candidate_counts)
                best_record_index = record_index
        if best is None:
            break
        _, features, distance, counts = best
        assert best_record_index is not None
        repeat_counts[best_record_index] += 1
        repeated_indexes.append(best_record_index)

    repeats_seen: Counter[int] = Counter()
    repeated_records: list[dict[str, Any]] = []
    for index in repeated_indexes:
        repeats_seen[index] += 1
        repeated = copy.deepcopy(records[index])
        repeated.setdefault("sampling", {}).update(
            {
                "balance_strategy": "repeat_hard_by_structural_val_fit",
                "balance_repeat": repeats_seen[index],
            }
        )
        repeated_records.append(repeated)
    selected = [*records, *repeated_records]
    write_accepted_jsonl(selected, output_path)
    atomic_json(
        output_path.with_suffix(".stats.json"),
        {
            "strategy": "repeat_hard_by_structural_val_fit",
            "input_records": len(records),
            "output_records": len(selected),
            "original_records_preserved": len(records),
            "eligible_hard": eligible_count,
            "original_first_attempt_easy": len(records) - eligible_count,
            "repeated_hard_records": len(repeated_records),
            "unique_hard_records_repeated": len(repeat_counts),
            "max_growth_fraction": max_growth_fraction,
            "max_repeats_per_record": max_repeats_per_record,
            "axis_weights": dict(zip(_BALANCE_AXES, weights, strict=True)),
            "weighted_tv_before": before_distance,
            "weighted_tv_after": distance,
            "input_counts": _serializable_axis_counts(before_counts),
            "output_counts": _serializable_axis_counts(counts),
            "reference_counts": _serializable_axis_counts(reference_counts),
            "reference_records": len(reference),
            "seed": seed,
        },
    )
    return output_path


def accept_rollout(
    rollout: Rollout,
    trace: AgentTrace,
    require_inspection: bool = True,
) -> bool:
    """Return whether a rollout satisfies every available native label."""

    if rollout.result is None:
        return False
    if trace.decisive_step is not None:
        if (
            not rollout.result.is_failure
            or not rollout.result.evidence_step_ids
            or _normalized_step(trace.decisive_step)
            != _normalized_step(rollout.result.evidence_step_ids[0])
        ):
            return False
        if trace.responsible_agents:
            if _normalized_agent(trace.responsible_agents[0]) != _normalized_agent(
                rollout.result.responsible_agent
            ):
                return False
        elif rollout.result.responsible_agent.strip():
            return False
    elif (
        rollout.result.is_failure != trace.is_failure
        or rollout.result.evidence_step_ids
        or rollout.result.responsible_agent.strip()
    ):
        return False
    return bool(rollout.inspected_step_ids) if require_inspection else True


def _normalized_step(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalized_agent(value: Any) -> str:
    return "".join(character.lower() for character in str(value or "") if character.isalnum())


def rejection_reason(rollout: Rollout, trace: AgentTrace, require_inspection: bool = True) -> str:
    if rollout.result is None:
        return (
            "no_prediction"
            if not rollout.error
            else f"rollout_error:{rollout.error.split(':', 1)[0]}"
        )
    if require_inspection and not rollout.inspected_step_ids:
        return "no_inspection"
    if trace.decisive_step is not None:
        if not rollout.result.is_failure:
            return "failure_label_mismatch"
        if not rollout.result.evidence_step_ids:
            return "missing_decisive_step"
        if _normalized_step(trace.decisive_step) != _normalized_step(
            rollout.result.evidence_step_ids[0]
        ):
            return "wrong_decisive_step"
        if trace.responsible_agents:
            if _normalized_agent(trace.responsible_agents[0]) != _normalized_agent(
                rollout.result.responsible_agent
            ):
                return "wrong_responsible_agent"
        elif rollout.result.responsible_agent.strip():
            return "unsupported_responsible_agent"
    elif rollout.result.is_failure != trace.is_failure:
        return "failure_label_mismatch"
    elif rollout.result.evidence_step_ids or rollout.result.responsible_agent.strip():
        return "unsupported_attribution"
    return "accepted"


def _has_final_answer_tool_call(messages: list[dict[str, Any]]) -> bool:
    return any(
        (tool_call.get("function") or {}).get("name") == "final_answer"
        for message in messages
        if message.get("role") == "assistant"
        for tool_call in (message.get("tool_calls") or [])
    )


def curated_record(
    rollout: Rollout,
    trace: AgentTrace,
    *,
    attempt: int,
    quality_tier: QualityTier,
    require_inspection: bool,
) -> tuple[dict[str, Any] | None, str]:
    if rollout.result is not None and not _has_final_answer_tool_call(rollout.messages):
        return None, "missing_final_answer_tool_call"
    if accept_rollout(rollout, trace, require_inspection=require_inspection):
        tier = "strict"
        messages = rollout.messages
        reason = "strict_correct"
    else:
        return None, rejection_reason(rollout, trace, require_inspection)
    try:
        record = accepted_record(
            sample_id=trace.sample_id,
            messages=messages,
            tools=tools_for_trace(trace),
            sampling={
                "attempt": attempt,
                "quality_tier": tier,
                "curation_reason": reason,
                "inspected_step_ids": rollout.inspected_step_ids,
            },
        )
    except ValueError:
        return None, "invalid_training_trajectory"
    return record, reason


def _sample_trace(
    trace: AgentTrace,
    model: ChatModel,
    attempts_per_item: int,
    accepted_per_item: int,
    max_turns: int,
    require_inspection: bool,
    harness_name: str,
    quality_tier: QualityTier,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def run_attempt(attempt: int) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        rollout = run_rollout(trace, model, max_turns=max_turns, harness_name=harness_name)
        record, reason = curated_record(
            rollout,
            trace,
            attempt=attempt,
            quality_tier=quality_tier,
            require_inspection=require_inspection,
        )
        diagnostic = {
            "sample_id": trace.sample_id,
            "score": float(record is not None),
            "reason": reason,
            "task_mode": trace.structured_summary()["task_mode"],
            "turns": rollout.turns,
            "inspected_step_ids": rollout.inspected_step_ids,
            "prompt_tokens": rollout.prompt_tokens,
            "completion_tokens": rollout.completion_tokens,
            "total_tokens": rollout.total_tokens,
            "token_usage_available": rollout.token_usage_available,
            "max_context_chars": rollout.max_context_chars,
            "cumulative_context_chars": rollout.cumulative_context_chars,
            "repeated_inspections": rollout.repeated_inspections,
            "prediction": asdict(rollout.result) if rollout.result is not None else None,
            "error": rollout.error,
            # Retain rejected trajectories for later offline re-curation.
            "messages": rollout.messages,
        }
        return record, diagnostic

    return policy_attempts(
        attempts_per_item=attempts_per_item,
        accepted_per_item=accepted_per_item,
        policy_attempt=run_attempt,
    )


def collect_accepted(
    traces: Iterable[AgentTrace],
    model: ChatModel,
    attempts_per_item: int = 3,
    accepted_per_item: int = 1,
    max_turns: int = 10,
    require_inspection: bool = True,
    harness_name: str = "initial_harness",
    quality_tier: QualityTier = "strict",
    concurrency: int = 1,
    skip_sample_ids: set[str] | None = None,
    on_accept: Callable[[dict[str, Any]], None] | None = None,
    on_attempt: Callable[[dict[str, Any]], None] | None = None,
    on_item_complete: Callable[[int, int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], SamplingStats]:
    if attempts_per_item < 1 or accepted_per_item < 1 or concurrency < 1:
        raise ValueError("attempts, accepted count, and concurrency must be positive")
    accepted: list[dict[str, Any]] = []
    stats = SamplingStats()
    progress = SamplingProgress(stats)
    skipped = skip_sample_ids or set()

    def consume(result: tuple[list[dict[str, Any]], list[dict[str, Any]]]) -> None:
        item_records, diagnostics = result
        for record in item_records:
            accepted.append(record)
            if on_accept is not None:
                on_accept(record)
        for diagnostic in progress.complete(item_records, diagnostics):
            if on_attempt is not None:
                on_attempt(diagnostic)
        if on_item_complete is not None:
            on_item_complete(stats.items, stats.attempts, stats.accepted)

    def eligible_traces():
        for trace in traces:
            if trace.sample_id in skipped:
                stats.skipped_existing += 1
                continue
            yield trace

    def sample_trace(trace: AgentTrace):
        return _sample_trace(
            trace,
            model,
            attempts_per_item,
            accepted_per_item,
            max_turns,
            require_inspection,
            harness_name,
            quality_tier,
        )

    for result in in_completion_order(eligible_traces(), sample_trace, concurrency):
        consume(result)
    return accepted, stats


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    write_accepted_jsonl(records, path)


def _policy_client(
    config: dict[str, Any],
    *,
    policy_model: str,
    policy_base_url: str,
    temperature: float | None,
) -> OpenAICompatibleClient:
    rollout = dict(config.get("rollout") or {})
    sampling = dict(config.get("rejection_sampling") or {})
    return OpenAICompatibleClient(
        model=policy_model,
        base_url=policy_base_url,
        temperature=float(
            temperature if temperature is not None else sampling.get("temperature", 0.7)
        ),
        top_p=rollout.get("top_p", 0.95),
        top_k=rollout.get("top_k", 20),
        max_tokens=int(rollout.get("max_tokens", 4096)),
        reasoning_effort=rollout.get("reasoning_effort", "medium"),
        enable_thinking=bool(rollout.get("enable_thinking", True)),
        timeout=float(rollout.get("timeout_seconds", 300)),
    )


def collect(request):
    """Collect exact policy successes."""
    from src.training.sampling import SamplingRecordWriter, read_accepted_jsonl

    config = load_config(request.config_path)
    sampling = config.get("rejection_sampling") or {}
    client = _policy_client(
        config,
        policy_model=request.policy_model,
        policy_base_url=request.base_url,
        temperature=request.temperature,
    )
    request.output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = request.output_dir / "accepted_rollouts.jsonl"
    existing = (
        read_accepted_jsonl(accepted_path) if request.resume and accepted_path.is_file() else []
    )
    with SamplingRecordWriter(
        accepted_path, request.output_dir / "sampling_attempts.jsonl", append=request.resume
    ) as writer:

        def on_attempt(record):
            record = {**record, "accepted_total": len(existing) + int(record["accepted_new"])}
            writer.write_attempt(record)
            if request.on_attempt:
                request.on_attempt(record)

        records, stats = collect_accepted(
            iter_traces(request.data_path, request.limit),
            client,
            attempts_per_item=request.attempts_per_item,
            accepted_per_item=request.accepted_per_item,
            max_turns=request.max_turns,
            require_inspection=bool(sampling.get("require_inspection", True)),
            harness_name=request.harness_name,
            quality_tier=request.quality_tier,
            concurrency=request.concurrency,
            skip_sample_ids={r["sample_id"] for r in existing},
            on_accept=writer.write_accepted,
            on_attempt=on_attempt,
        )
    payload = {
        **asdict(stats),
        "existing_records": len(existing),
        "total_accepted_records": len(existing) + len(records),
        "attempts_per_item": request.attempts_per_item,
        "concurrency": request.concurrency,
        "quality_tier": request.quality_tier,
        "repair_strategy": "none",
    }
    atomic_json(request.output_dir / "sampling_stats.json", payload)
    return stats


def prepare_training_source(request):
    settings = dict(request.sampling_config or {})
    if not settings.get("balance_by_val_difficulty", False):
        return request.accepted_path
    reference = Path(str(settings.get("balance_reference_data") or ""))
    if not reference.is_absolute():
        reference = Path.cwd() / reference
    if not reference.is_file():
        raise FileNotFoundError(
            f"Who&When balance reference does not exist: {reference}"
        )
    return balance_training_rollouts(
        accepted_path=request.accepted_path,
        data_path=request.data_path,
        reference_path=reference,
        output_path=request.output_dir / "balanced_rollouts.jsonl",
        seed=int(settings.get("balance_seed", 42)),
        max_growth_fraction=float(settings.get("balance_max_growth_fraction", 0.5)),
        max_repeats_per_record=int(settings.get("balance_max_repeats_per_record", 3)),
        axis_weights=dict(settings.get("balance_axis_weights") or {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/who_when.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--attempts-per-item", type=int)
    parser.add_argument("--accepted-per-item", type=int, default=1)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--quality-tier", choices=("strict", "balanced"))
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--harness")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-no-inspection", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    selected_harness = harness_spec(
        config,
        project_root=Path(__file__).resolve().parents[3],
    )
    if selected_harness.kind != "failure_detection":
        parser.error(
            "this rejection sampler consumes Who&When labeled trajectories; "
            f"domain={selected_harness.domain} must use its own sampler"
        )
    harness_name = args.harness or active_harness(config)
    sampling = dict(config.get("rejection_sampling") or {})
    client = _policy_client(
        config,
        policy_model=args.model,
        policy_base_url=args.base_url,
        temperature=args.temperature,
    )
    records, stats = collect_accepted(
        iter_traces(args.data, args.limit),
        client,
        attempts_per_item=args.attempts_per_item or int(sampling.get("attempts_per_item", 3)),
        accepted_per_item=args.accepted_per_item,
        max_turns=args.max_turns,
        require_inspection=not args.allow_no_inspection,
        harness_name=harness_name,
        quality_tier=args.quality_tier or str(sampling.get("quality_tier", "strict")),
        concurrency=args.concurrency or int(sampling.get("concurrency", 1)),
    )
    write_jsonl(records, args.output)
    print(json.dumps(asdict(stats), sort_keys=True))


if __name__ == "__main__":
    main()
