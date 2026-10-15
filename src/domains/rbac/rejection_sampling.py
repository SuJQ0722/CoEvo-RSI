"""Binary rejection sampling for the RBAC generative fallback policy."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.config import load_config
from src.domains.rbac.data import RBACSample, load, load_policy_rows
from src.reject_sampling import SamplingProgress, policy_attempts
from src.runner.execution import in_completion_order
from src.runner.model_client import OpenAICompatibleClient
from src.training.message_schema import assistant_tool_message, tool_result_message
from src.training.sampling import (
    SamplingRecordWriter,
    SamplingStats,
    accepted_record,
    read_accepted_jsonl,
    write_accepted_jsonl,
)
from src.utils.persistence import atomic_json
from harnesses.rbac.loader import build_harness

try:
    from src.domains.rbac.runtime.exec_eval import (
        exec_match_safe,  # type: ignore[import-untyped]
    )
except ImportError:
    exec_match_safe = None  # type: ignore[assignment]


def _resolve_db_path(db_dir: str | None, db_id: str) -> str | None:
    from src.domains.rbac.databases import resolve_db_path

    return resolve_db_path(db_dir, db_id)


def _final_access_label(record: dict[str, Any]) -> int:
    """Return the label submitted by a canonical accepted trajectory."""
    for message in reversed(record.get("messages") or []):
        if message.get("role") != "assistant":
            continue
        for tool_call in reversed(message.get("tool_calls") or []):
            function = tool_call.get("function") or {}
            if function.get("name") != "final_answer":
                continue
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            value = arguments.get("access_granted")
            if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
                value = value.strip().lower() == "true"
            if isinstance(value, bool):
                return int(value)
    raise ValueError(f"accepted record has no final access label: {record.get('sample_id')}")


def _gold_record(
    sample: RBACSample,
    harness: Any,
    *,
    quality_tier: str = "gold",
    curation_reason: str | None = None,
    attempt: int = 0,
) -> dict[str, Any]:
    if sample.valid and not sample.gold_sql:
        raise ValueError(f"allowed gold sample has no SQL: {sample.sample_id}")
    route = SimpleNamespace(
        decision=None, source="gold_training_data", confidence=1.0, required_role=None
    )
    messages: list[dict[str, Any]] = list(
        harness.messages(sample.instruction, sample.test_role, route)
    )
    arguments: dict[str, Any] = {"access_granted": bool(sample.valid)}
    if sample.valid:
        arguments["sql"] = sample.gold_sql
    call_id = f"gold-final-{sample.sample_id.replace(':', '-')}"
    call = {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "final_answer",
            "arguments": json.dumps(arguments),
        },
    }
    messages.extend(
        [
            assistant_tool_message([call]),
            tool_result_message(
                call,
                _submitted_tool_content(
                    sample, sample.gold_sql if sample.valid else None
                ),
            ),
        ]
    )
    return accepted_record(
        sample_id=sample.sample_id,
        messages=messages,
        tools=harness.tools,
        sampling={
            "attempt": attempt,
            "score": 1,
            "quality_tier": quality_tier,
            "curation_reason": curation_reason
            or ("original_gold_allow" if sample.valid else "original_gold_deny"),
            "sql": sample.gold_sql if sample.valid else None,
        },
        metadata={"source": "data/rbac original-derived training split"},
    )


def _tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any] | None:
    value = (tool_call.get("function") or {}).get("arguments")
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return dict(parsed) if isinstance(parsed, dict) else None
    return None


def _normalized_access(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    # JSON-schema providers occasionally serialize booleans as 0/1. Treat only
    # those exact integers as a formatting error; arbitrary truthy values are
    # not safe to repair.
    if type(value) is int and value in {0, 1}:
        return bool(value)
    return None


def _submitted_tool_content(sample: RBACSample, sql: str | None) -> str:
    return json.dumps(
        {
            "status": "submitted",
            "result": "ACCESS_GRANTED" if sample.valid else "ACCESS_DENIED",
            "sql": sql if sample.valid else None,
        }
    )


def _format_only_repair(
    sample: RBACSample,
    messages: list[dict[str, Any]],
    *,
    db_path: str | None,
) -> tuple[list[dict[str, Any]], str | None] | None:
    """Canonicalize a malformed final call only when its intent is provably gold."""
    repaired = deepcopy(messages)
    for message_index in range(len(repaired) - 1, -1, -1):
        message = repaired[message_index]
        if message.get("role") != "assistant":
            continue
        for tool_call in reversed(message.get("tool_calls") or []):
            function = tool_call.get("function") or {}
            if function.get("name") != "final_answer":
                continue
            arguments = _tool_arguments(tool_call)
            if arguments is None:
                return None
            access = _normalized_access(arguments.get("access_granted"))
            if access is None or int(access) != sample.valid:
                return None
            raw_sql = arguments.get("sql")
            sql = raw_sql.strip() if isinstance(raw_sql, str) else None
            if sample.valid:
                if not (
                    exec_match_safe is not None
                    and db_path
                    and sql
                    and sample.gold_sql
                    and exec_match_safe(db_path, sql, sample.gold_sql)
                ):
                    return None
            else:
                sql = None
            canonical = {"access_granted": bool(sample.valid)}
            if sql:
                canonical["sql"] = sql
            function["arguments"] = (
                canonical
                if isinstance(function.get("arguments"), dict)
                else json.dumps(canonical, ensure_ascii=False)
            )
            call_id = str(tool_call.get("id") or "")
            if not call_id:
                return None
            repaired = repaired[: message_index + 1]
            repaired.append(
                tool_result_message(tool_call, _submitted_tool_content(sample, sql))
            )
            return repaired, sql
    return None


def _verified_prefix_repair(
    sample: RBACSample,
    messages: list[dict[str, Any]],
    *,
    db_path: str | None,
) -> tuple[list[dict[str, Any]], str] | None:
    """Keep a prefix through a gold-equivalent test_sql and replace only its suffix."""
    if not (
        sample.valid
        and sample.gold_sql
        and db_path
        and exec_match_safe is not None
    ):
        return None
    trusted: tuple[int, dict[str, Any], str] | None = None
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            if function.get("name") != "test_sql":
                continue
            arguments = _tool_arguments(tool_call)
            sql = (arguments or {}).get("sql")
            if not isinstance(sql, str) or not sql.strip():
                continue
            call_id = str(tool_call.get("id") or "")
            if not call_id:
                continue
            result_index = next(
                (
                    candidate
                    for candidate in range(index + 1, len(messages))
                    if messages[candidate].get("role") == "tool"
                    and str(messages[candidate].get("tool_call_id") or "") == call_id
                ),
                None,
            )
            if result_index is None:
                continue
            try:
                result_payload = json.loads(str(messages[result_index].get("content") or ""))
            except json.JSONDecodeError:
                continue
            if not isinstance(result_payload, dict) or result_payload.get("error"):
                continue
            if exec_match_safe(db_path, sql.strip(), sample.gold_sql):
                trusted = (result_index, tool_call, sql.strip())
    if trusted is None:
        return None
    result_index, _, sql = trusted
    call = {
        "id": f"repaired-final-{sample.sample_id.replace(':', '-')}",
        "type": "function",
        "function": {
            "name": "final_answer",
            "arguments": json.dumps(
                {"access_granted": True, "sql": sql}, ensure_ascii=False
            ),
        },
    }
    repaired = deepcopy(messages[: result_index + 1])
    repaired.extend(
        [
            assistant_tool_message([call]),
            tool_result_message(call, _submitted_tool_content(sample, sql)),
        ]
    )
    return repaired, sql


def _quality_repair_record(
    sample: RBACSample,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    attempt: int,
    db_path: str | None,
) -> tuple[dict[str, Any], str] | None:
    repaired = _format_only_repair(sample, messages, db_path=db_path)
    reason = "format_only_final_repair"
    if repaired is None:
        repaired = _verified_prefix_repair(sample, messages, db_path=db_path)
        reason = "verified_test_sql_suffix_repair"
    if repaired is None:
        return None
    repaired_messages, sql = repaired
    try:
        record = accepted_record(
            sample_id=sample.sample_id,
            messages=repaired_messages,
            tools=tools,
            sampling={
                "attempt": attempt,
                "score": 1,
                "quality_tier": "verified_repaired",
                "curation_reason": reason,
                "sql": sql,
            },
        )
    except ValueError:
        return None
    return record, reason


def build_balanced_training_rollouts(
    *,
    accepted_path: Path,
    data_path: Path,
    output_path: Path,
    harness_name: str,
    min_deny_to_allow_ratio: float = 1.0,
    max_deny_to_allow_ratio: float = 2.0,
    target_deny_to_allow_ratio: float = 1.0,
    seed: int = 42,
    strategy: str = "supplement_gold",
) -> dict[str, Any]:
    """Balance verified rollouts by deterministic downsampling or gold supplements."""
    if not (
        0 < min_deny_to_allow_ratio
        <= target_deny_to_allow_ratio
        <= max_deny_to_allow_ratio
    ):
        raise ValueError("RBAC deny:allow ratio must satisfy 0 < min <= target <= max")
    if strategy not in {"downsample", "supplement_gold"}:
        raise ValueError("RBAC balance strategy must be downsample or supplement_gold")

    accepted = read_accepted_jsonl(accepted_path)
    accepted_tiers = Counter(
        str((record.get("sampling") or {}).get("quality_tier") or "strict") for record in accepted
    )
    counts = Counter(_final_access_label(record) for record in accepted)
    supplements: list[dict[str, Any]] = []
    dropped_for_balance = 0
    if strategy == "downsample":
        if not counts[0] or not counts[1]:
            raise RuntimeError(
                "cannot downsample RBAC training data when one label has no verified records"
            )
        keep = {0: counts[0], 1: counts[1]}
        if counts[0] > target_deny_to_allow_ratio * counts[1]:
            keep[0] = math.floor(target_deny_to_allow_ratio * counts[1])
        elif counts[0] < target_deny_to_allow_ratio * counts[1]:
            keep[1] = math.floor(counts[0] / target_deny_to_allow_ratio)
        selected_indices: set[int] = set()
        for label in (0, 1):
            ranked = sorted(
                (
                    (index, record)
                    for index, record in enumerate(accepted)
                    if _final_access_label(record) == label
                ),
                key=lambda item: hashlib.sha256(
                    (
                        f"{seed}|rbac-downsample|{item[0]}|"
                        f"{item[1].get('sample_id')}"
                    ).encode()
                ).hexdigest(),
            )
            selected_indices.update(index for index, _ in ranked[: keep[label]])
        records = [
            record for index, record in enumerate(accepted) if index in selected_indices
        ]
        dropped_for_balance = len(accepted) - len(records)
    else:
        # supplement_gold: rejection sampling alone may not yield enough verified,
        # high-quality records for the minority label. Top up from the TRAINING
        # split's own gold SQL/decisions (never validation or test) to obtain
        # more high-quality supervision at the target deny:allow ratio.
        desired = dict(counts)
        if counts[0] > target_deny_to_allow_ratio * counts[1]:
            desired[1] = math.ceil(counts[0] / target_deny_to_allow_ratio)
        elif counts[0] < target_deny_to_allow_ratio * counts[1]:
            desired[0] = math.ceil(target_deny_to_allow_ratio * counts[1])

        source_samples = list(load(data_path))
        harness = build_harness(harness_name, load_policy_rows(data_path))
        existing = {
            (str(record.get("sample_id")), _final_access_label(record))
            for record in accepted
        }
        for label in (0, 1):
            needed = desired.get(label, 0) - counts.get(label, 0)
            candidates = [
                sample
                for sample in source_samples
                if sample.valid == label and (sample.sample_id, label) not in existing
            ]
            candidates.sort(
                key=lambda sample: hashlib.sha256(
                    f"{seed}|rbac-gold-balance|{sample.sample_id}".encode()
                ).hexdigest()
            )
            if len(candidates) < needed:
                raise RuntimeError(
                    f"not enough original RBAC label={label} samples to add {needed}; "
                    f"only {len(candidates)} are available"
                )
            supplements.extend(
                _gold_record(sample, harness) for sample in candidates[:needed]
            )
        records = [*accepted, *supplements]
    final_counts = Counter(_final_access_label(record) for record in records)
    if not final_counts[1]:
        raise RuntimeError("balanced RBAC training data contains no allowed samples")
    ratio = final_counts[0] / final_counts[1]
    if not min_deny_to_allow_ratio <= ratio <= max_deny_to_allow_ratio:
        raise RuntimeError(
            f"RBAC training deny:allow ratio {ratio:.6f} is outside "
            f"[{min_deny_to_allow_ratio}, {max_deny_to_allow_ratio}]"
        )
    write_accepted_jsonl(records, output_path)
    report = {
        "accepted_records": len(accepted),
        "strict_accepted_records": accepted_tiers["strict"],
        "repaired_accepted_records": sum(
            count for tier, count in accepted_tiers.items() if tier != "strict"
        ),
        "gold_supplements": len(supplements),
        "dropped_for_balance": dropped_for_balance,
        "balance_strategy": strategy,
        "allow": final_counts[1],
        "deny": final_counts[0],
        "deny_to_allow_ratio": ratio,
        "target_deny_to_allow_ratio": target_deny_to_allow_ratio,
        "source_data": str(data_path),
    }
    atomic_json(output_path.parent / "training_mix.json", report)
    return report


def _score_attempt(
    sample: RBACSample,
    result: Any,
    harness: Any,
    *,
    db_path: str | None,
    require_sql_execution: bool,
) -> tuple[int, int, str | None, str]:
    error = str(result.error or "")
    predicted = int(harness.parse_access(result.response)) if not error else 0
    predicted_sql = getattr(result, "sql", None)
    if error:
        return 0, predicted, predicted_sql, "error"
    if predicted != sample.valid:
        return 0, predicted, predicted_sql, "binary_mismatch"
    if sample.valid == 0:
        return 1, predicted, predicted_sql, "binary_match_denied"
    if exec_match_safe is not None and db_path and predicted_sql and sample.gold_sql:
        if exec_match_safe(db_path, predicted_sql, sample.gold_sql):
            return 1, predicted, predicted_sql, "binary_match_exec_verified"
        return 0, predicted, predicted_sql, "exec_mismatch"
    if require_sql_execution:
        return 0, predicted, predicted_sql, "sql_verification_unavailable"
    return 1, predicted, predicted_sql, "binary_match_allowed"


def collect_to_directory(
    *,
    config_path: Path,
    data_path: Path,
    output_dir: Path,
    policy_model: str,
    base_url: str,
    harness_name: str,
    attempts_per_item: int,
    accepted_per_item: int,
    concurrency: int,
    quality_tier: str = "strict",
    repair_strategy: str | None = None,
    require_sql_execution: bool | None = None,
    limit: int | None = None,
    resume: bool = False,
    db_dir: str | None = None,
    temperature: float | None = None,
    on_attempt: Callable[[dict[str, Any]], None] | None = None,
) -> SamplingStats:
    """Keep only exact 0/1 access matches; for allowed samples also check SQL via exec eval."""
    if quality_tier not in {"strict", "balanced"}:
        raise ValueError(f"unsupported RBAC quality tier: {quality_tier!r}")
    config = load_config(config_path)
    project_root = Path(__file__).resolve().parents[3]
    train_value = (config.get("data") or {}).get("splits", {}).get("train")
    train_path = Path(str(train_value))
    if not train_path.is_absolute():
        train_path = project_root / train_path
    if not train_path.is_file():
        # Keep standalone sampling usable with an explicitly supplied dataset;
        # production configs still use their dedicated policy-training split.
        train_path = data_path
    harness = build_harness(harness_name, load_policy_rows(train_path))
    settings = dict(config.get("rejection_sampling") or {})
    rollout = dict(config.get("rollout") or {})
    repair_strategy = repair_strategy or str(
        settings.get("repair_strategy", "gold_final")
    )
    if repair_strategy not in {"gold_final", "verified_only", "none"}:
        raise ValueError(
            "RBAC repair_strategy must be gold_final, verified_only, or none"
        )
    if require_sql_execution is None:
        require_sql_execution = bool(settings.get("require_sql_execution", False))
    model = OpenAICompatibleClient(
        model=policy_model,
        base_url=base_url,
        temperature=float(
            temperature if temperature is not None else settings.get("temperature", 0.7)
        ),
        top_p=float(rollout.get("top_p", 1.0)),
        top_k=rollout.get("top_k"),
        max_tokens=int(rollout.get("max_tokens", 512)),
        reasoning_effort=rollout.get("reasoning_effort"),
        enable_thinking=rollout.get("enable_thinking", False),
        timeout=float(rollout.get("timeout_seconds", 300)),
    )
    # Resolve db_dir from standard locations if not provided
    if not db_dir:
        for candidate in ["data/spider/database", "data/spider"]:
            p = project_root / candidate
            if p.is_dir():
                db_dir = str(p)
                break

    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / "accepted_rollouts.jsonl"
    attempts_path = output_dir / "sampling_attempts.jsonl"
    existing: list[dict[str, Any]] = []
    if resume and accepted_path.is_file():
        existing = read_accepted_jsonl(accepted_path)
    counts = Counter(str(item.get("sample_id")) for item in existing)
    all_samples = list(load(data_path, limit=limit))
    samples = [sample for sample in all_samples if counts[sample.sample_id] < accepted_per_item]
    stats = SamplingStats(skipped_existing=len(all_samples) - len(samples))
    fallback = SimpleNamespace(
        decision=None, source="rejection_sampling", confidence=0.0, required_role=None
    )

    def sample_one(sample: RBACSample) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        needed = accepted_per_item - counts[sample.sample_id]
        sample_db_path = _resolve_db_path(db_dir, sample.db_id)
        gold_sql = getattr(sample, "gold_sql", None)
        completed_attempts = 0
        repair_candidate: tuple[dict[str, Any], str, int] | None = None

        def policy_attempt(
            attempt: int,
        ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
            nonlocal completed_attempts, repair_candidate
            result = harness.call_llm(
                model,
                sample.instruction,
                sample.test_role,
                fallback,
                db_path=sample_db_path,
            )
            error = result.error
            if not error:
                completed_attempts += 1
            score, predicted, predicted_sql, reason = _score_attempt(
                sample,
                result,
                harness,
                db_path=sample_db_path,
                require_sql_execution=require_sql_execution,
            )

            accepted_item: dict[str, Any] | None = None
            trajectory_error = ""
            if score == 1:
                try:
                    accepted_item = accepted_record(
                        sample_id=sample.sample_id,
                        messages=result.messages,
                        tools=harness.tools,
                        sampling={
                            "attempt": attempt,
                            "score": 1,
                            "quality_tier": "strict",
                            "curation_reason": reason,
                            "sql": predicted_sql,
                        },
                    )
                except ValueError as exc:
                    score = 0
                    reason = "invalid_training_trajectory"
                    trajectory_error = str(exc)

            diagnostic = {
                "sample_id": sample.sample_id,
                "attempt": attempt,
                "score": score,
                "accepted": bool(score),
                "predicted": predicted,
                "gold": sample.valid,
                "predicted_sql": predicted_sql,
                "gold_sql": gold_sql,
                "reason": reason,
                "error": trajectory_error or error,
                "total_tokens": (getattr(result, "usage", {}) or {}).get("total_tokens"),
                "token_usage_available": bool(getattr(result, "usage", {})),
                # Persist rejected trajectories so stricter offline curation
                # does not require another policy rollout.
                "messages": result.messages,
            }
            if accepted_item is None and (
                quality_tier == "balanced"
                and repair_strategy == "verified_only"
                and repair_candidate is None
            ):
                repaired = _quality_repair_record(
                    sample,
                    result.messages,
                    harness.tools,
                    attempt=attempt,
                    db_path=sample_db_path,
                )
                if repaired is not None:
                    record, repair_reason = repaired
                    # The shared orchestrator appends this policy diagnostic
                    # after the callback returns.
                    repair_candidate = (record, repair_reason, attempt - 1)
                    diagnostic["repair_candidate"] = repair_reason
            return accepted_item, diagnostic

        accepted, diagnostics = policy_attempts(
            attempts_per_item=attempts_per_item,
            accepted_per_item=needed,
            policy_attempt=policy_attempt,
        )
        if not accepted and repair_candidate is not None:
            record, repair_reason, diagnostic_index = repair_candidate
            accepted.append(record)
            diagnostic = diagnostics[diagnostic_index]
            diagnostic["original_reason"] = diagnostic["reason"]
            diagnostic["reason"] = repair_reason
            diagnostic["score"] = 1
            diagnostic["accepted"] = True
            diagnostic["trajectory_repaired"] = True
        elif (
            not accepted
            and quality_tier == "balanced"
            and repair_strategy == "gold_final"
            and completed_attempts
        ):
            accepted.append(
                _gold_record(
                    sample,
                    harness,
                    quality_tier="balanced_repaired",
                    curation_reason="gold_answer_repair",
                    attempt=len(diagnostics),
                )
            )
            if diagnostics:
                diagnostics[-1]["gold_repaired"] = True
        return accepted, diagnostics

    progress = SamplingProgress(stats, existing_records=len(existing))

    def consume(
        result: tuple[list[dict[str, Any]], list[dict[str, Any]]],
        writer: SamplingRecordWriter,
    ) -> None:
        accepted, diagnostics = result
        for record in accepted:
            writer.write_accepted(record)
        for diagnostic in progress.complete(accepted, diagnostics):
            writer.write_attempt(diagnostic)
            if on_attempt:
                on_attempt(diagnostic)

    with SamplingRecordWriter(accepted_path, attempts_path, append=resume) as writer:
        for result in in_completion_order(samples, sample_one, concurrency):
            # Persist each completed sample immediately so long RBAC runs expose
            # live accepted/attempt progress and remain resumable.
            consume(result, writer)
    stats_payload = asdict(stats)
    stats_payload["existing_records"] = len(existing)
    stats_payload["total_accepted_records"] = len(existing) + stats.accepted
    stats_payload["attempts_per_item"] = attempts_per_item
    stats_payload["accepted_per_item"] = accepted_per_item
    stats_payload["concurrency"] = concurrency
    stats_payload["quality_tier"] = quality_tier
    stats_payload["repair_strategy"] = repair_strategy
    stats_payload["require_sql_execution"] = require_sql_execution
    atomic_json(output_dir / "sampling_stats.json", stats_payload)
    return stats


def collect(request):
    options = request.kwargs()
    options.pop("max_turns")
    return collect_to_directory(**options)


def prepare_training_source(request):
    if request.quality_tier != "balanced":
        return request.accepted_path
    settings = request.sampling_config
    output = request.output_dir / "balanced_rollouts.jsonl"
    build_balanced_training_rollouts(
        accepted_path=request.accepted_path,
        data_path=request.data_path,
        output_path=output,
        harness_name=request.harness_name,
        min_deny_to_allow_ratio=float(settings.get("min_deny_to_allow_ratio", 1.0)),
        max_deny_to_allow_ratio=float(settings.get("max_deny_to_allow_ratio", 2.0)),
        target_deny_to_allow_ratio=float(settings.get("target_deny_to_allow_ratio", 1.0)),
        seed=int(settings.get("balance_seed", 42)),
        strategy=str(settings.get("balance_strategy", "supplement_gold")),
    )
    return output
