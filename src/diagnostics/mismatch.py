"""Offline interface diagnostics consumed by model-selection ranking."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from jsonschema import validators
from jsonschema.exceptions import SchemaError
from referencing import Registry
from referencing.exceptions import Unresolvable

from src.domains.io import read_records

from .observe import TRACE_VERSION

METRIC_VERSION = "model_harness_mismatch_v3"
RATE_NAMES = (
    "tool_name_error_rate",
    "tool_argument_error_rate",
    "argument_json_error_rate",
    "response_protocol_error_rate",
    "tool_return_error_rate",
    "final_format_error_rate",
    "ever_final_format_error_rate",
    "protocol_mismatch_turn_rate",
    "argument_mismatch_turn_rate",
    "final_format_mismatch_turn_rate",
    "interface_mismatch_turn_rate",
    "interface_mismatch_task_rate",
    "transport_repair_rate",
    "interface_recovery_rate",
    "unfinished_task_rate",
)


class MismatchContract:
    """Override domain semantics without adding domain branches to the controller.

    None means unobserved/unspecified, never an inferred success. Return errors
    are classified only at domain-defined envelope locations.
    """

    min_tool_calls = None
    max_tool_calls = None
    terminal_tool = None

    def state(self, record):
        return {"finished": None, "final_format_error": None, "rule_only": False}

    def tool_return_errors(self, name, payload):
        return None  # No known return contract; not a zero-error observation.

    def task_correct(self, result):
        return None


def _schemas(tools):
    if not isinstance(tools, list):
        return None
    schemas = {}
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            return None
        name = function["name"]
        if not name or name in schemas:
            return None
        schemas[name] = function.get("parameters")
    return schemas


def _argument_errors(value, schema):
    try:
        args = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return ["invalid_json"], True
    if not isinstance(args, dict):
        return ["not_object"], True
    if schema is None:
        return [], False
    try:
        validator = validators.validator_for(schema)
        validator.check_schema(schema)
        # An empty registry refuses remote references instead of fetching them.
        errors = list(validator(schema, registry=Registry()).iter_errors(args))
    except (SchemaError, Unresolvable, TypeError, ValueError, RecursionError):
        return [], False
    names = {
        "required": "missing_required",
        "type": "wrong_type",
        "additionalProperties": "extra_property",
    }
    return sorted({names.get(e.validator, "enum_or_bound") for e in errors}), True


def _rate(count, denominator):
    return {
        "count": count,
        "denominator": denominator,
        "rate": count / denominator if denominator else None,
        "reason": None if denominator else "no_observed_opportunities",
    }


def analyze_record(record, contract, quality=None):
    counts, denominators = Counter(), Counter()
    events, ids = [], set()
    state = contract.state(record)
    trace = record.get("interface_trace") or {}
    observed = trace.get("version") == TRACE_VERSION and isinstance(trace.get("requests"), list)
    requests = (
        trace["requests"]
        if observed
        else [
            {"response": m}
            for m in record.get("messages", [])
            if isinstance(m, dict) and m.get("role") == "assistant"
        ]
    )
    # New traces persist the post-client-repair response directly. Historical
    # traces only have transport_repaired=true; when their runtime transcript
    # aligns exactly with observed responses, it supplies the same effective
    # interface for accurate offline recomputation.
    response_indexes = [
        index for index, request in enumerate(requests) if "response" in request
    ]
    assistant_messages = [
        message
        for message in record.get("messages", [])
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    historical_effective = (
        dict(zip(response_indexes, assistant_messages, strict=True))
        if len(response_indexes) == len(assistant_messages)
        else {}
    )
    runtime_return_positions = {}
    if historical_effective:
        response_position = iter(response_indexes)
        pending_calls = []
        for message_index, message in enumerate(record.get("messages", [])):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                request_index = next(response_position)
                pending_calls = [
                    (
                        call.get("id"),
                        (call.get("function") or {}).get("name"),
                        request_index,
                    )
                    for call in message.get("tool_calls") or []
                    if isinstance(call, dict)
                ]
            elif message.get("role") == "tool":
                for call_index, (call_id, name, request_index) in enumerate(pending_calls):
                    if call_id == message.get("tool_call_id") and name == message.get("name"):
                        runtime_return_positions[message_index] = request_index
                        pending_calls.pop(call_index)
                        break
    final_attempts, final_bad, final_request = 0, False, None
    ever_final_bad = False
    mismatch = False
    mismatch_turns = {
        "protocol": set(),
        "argument": set(),
        "final_format": set(),
        "all": set(),
    }
    last_error_request = -1
    call_requests = {}
    full_checks = observed
    responses = calls_total = name_checks = argument_checks = 0

    def event(
        code,
        request=None,
        call=None,
        name=None,
        *,
        model_error=False,
        mismatch_category=None,
    ):
        nonlocal mismatch, last_error_request
        events.append({"code": code, "request_index": request, "call_index": call, "tool": name})
        if model_error:
            mismatch = True
            if isinstance(request, int) and 0 <= request < len(requests):
                mismatch_turns["all"].add(request)
                if mismatch_category is not None:
                    mismatch_turns[mismatch_category].add(request)
            last_error_request = max(
                last_error_request, request if request is not None else len(requests)
            )

    for ri, request in enumerate(requests):
        if "response" not in request:
            continue
        responses += 1
        denominators["response_protocol_error_rate"] += 1
        repaired = bool(request.get("transport_repaired"))
        if observed:
            denominators["transport_repair_rate"] += 1
            counts["transport_repair_rate"] += repaired
        effective = request.get("effective_response") if repaired else None
        if repaired and not isinstance(effective, dict):
            effective = historical_effective.get(ri)
        message = effective if isinstance(effective, dict) else request["response"]
        protocol_bad = False
        if not isinstance(message, dict):
            message = {}
            protocol_bad = True
        if message.get("role", "assistant") != "assistant":
            protocol_bad = True
        if message.get("content") is not None and not isinstance(message["content"], str):
            protocol_bad = True
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            calls = []
            protocol_bad = True
        if (contract.min_tool_calls is not None and len(calls) < contract.min_tool_calls) or (
            contract.max_tool_calls is not None and len(calls) > contract.max_tool_calls
        ):
            protocol_bad = True
        if repaired and not isinstance(effective, dict):
            # A legacy trace proves the client returned a changed tool-call
            # interface but does not retain it. Do not punish a successful
            # repair based only on the pre-repair provider representation.
            protocol_bad = False
        schemas = _schemas(request.get("tools"))
        if schemas is None:
            full_checks = False
            if observed:
                event("harness_schema_unavailable", ri)
        for ci, call in enumerate(calls):
            calls_total += 1
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                function = {}
                protocol_bad = True
            name = function.get("name")
            cid = call.get("id") if isinstance(call, dict) else None
            if isinstance(cid, str) and cid:
                call_requests.setdefault(cid, []).append(ri)
            if observed and not repaired:
                # Calls reconstructed from provider text receive client-owned
                # IDs. Their presence/uniqueness is a transport concern, not a
                # model–harness mismatch signal.
                if not isinstance(cid, str) or not cid or cid in ids:
                    protocol_bad = True
                    event("tool_call_id_error", ri, ci, name)
                if isinstance(cid, str):
                    ids.add(cid)
            if not isinstance(name, str) or not name:
                protocol_bad = True
                name = None
            if isinstance(call, dict) and call.get("type", "function") != "function":
                protocol_bad = True
            bad = False
            json_errors, _ = _argument_errors(function.get("arguments"), None)
            denominators["argument_json_error_rate"] += 1
            counts["argument_json_error_rate"] += bool(json_errors)
            if schemas is not None:
                name_checks += 1
                denominators["tool_name_error_rate"] += 1
                if name not in schemas:
                    bad = True
                    counts["tool_name_error_rate"] += 1
                    event(
                        "unknown_tool",
                        ri,
                        ci,
                        name,
                        model_error=True,
                        mismatch_category="protocol",
                    )
                else:
                    errors, checked = _argument_errors(function.get("arguments"), schemas[name])
                    if checked:
                        argument_checks += 1
                        denominators["tool_argument_error_rate"] += 1
                        counts["tool_argument_error_rate"] += bool(errors)
                        bad = bool(errors)
                        for code in errors:
                            event(
                                code,
                                ri,
                                ci,
                                name,
                                model_error=True,
                                mismatch_category="argument",
                            )
                    else:
                        full_checks = False
                        event("harness_parameter_schema_unavailable", ri, ci, name)
            elif json_errors:
                bad = True
                for code in json_errors:
                    event(
                        code,
                        ri,
                        ci,
                        name,
                        model_error=True,
                        mismatch_category="argument",
                    )
            if contract.terminal_tool and name == contract.terminal_tool:
                final_attempts += 1
                final_bad = bad or protocol_bad
                ever_final_bad |= final_bad
                final_request = ri
        if protocol_bad:
            counts["response_protocol_error_rate"] += 1
            event(
                "response_protocol_error",
                ri,
                model_error=True,
                mismatch_category="protocol",
            )
            if final_request == ri:
                final_bad = ever_final_bad = True

    tool_returns = return_checks = 0
    return_categories = Counter()
    for message_index, message in enumerate(record.get("messages", [])):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        tool_returns += 1
        try:
            payload = json.loads(message.get("content", ""))
        except (TypeError, ValueError):
            payload = message.get("content")
        categories = contract.tool_return_errors(message.get("name"), payload)
        if categories is None:
            continue
        return_checks += 1
        denominators["tool_return_error_rate"] += 1
        counts["tool_return_error_rate"] += bool(categories)
        positions = call_requests.get(message.get("tool_call_id"), [])
        position = runtime_return_positions.get(message_index)
        if position is None:
            position = positions[0] if len(positions) == 1 else None
        for category in sorted(set(categories)):
            return_categories[category] += 1
            event(
                "tool_return_" + category,
                position,
                name=message.get("name"),
                model_error=category == "argument_semantic",
                mismatch_category="argument" if category == "argument_semantic" else None,
            )

    finished = state.get("finished")
    format_bad = state.get("final_format_error")
    if final_attempts and (full_checks or format_bad is not None):
        final_bad = final_bad or bool(format_bad)
        ever_final_bad |= final_bad
        denominators["final_format_error_rate"] = 1
        counts["final_format_error_rate"] = int(final_bad)
        denominators["ever_final_format_error_rate"] = 1
        counts["ever_final_format_error_rate"] = int(ever_final_bad)
        if final_bad:
            event(
                "final_format_error",
                final_request,
                model_error=True,
                mismatch_category="final_format",
            )
            finished = False
    if responses:
        # Primary IMR is the union of the three mismatch categories at the
        # assistant-turn level. A turn with several violations counts once.
        # Only fully observed tasks contribute turns to this comparable rate.
        if full_checks and contract.min_tool_calls is not None:
            turn_rates = {
                "protocol_mismatch_turn_rate": mismatch_turns["protocol"],
                "argument_mismatch_turn_rate": mismatch_turns["argument"],
                "final_format_mismatch_turn_rate": mismatch_turns["final_format"],
                "interface_mismatch_turn_rate": mismatch_turns["all"],
            }
            for name, turn_indexes in turn_rates.items():
                denominators[name] = responses
                counts[name] = len(turn_indexes)
            # Retain the old task-level union as a secondary compatibility
            # diagnostic; selection and the IMR label use the turn rate above.
            denominators["interface_mismatch_task_rate"] = 1
            counts["interface_mismatch_task_rate"] = int(mismatch)
        if finished is not None:
            denominators["unfinished_task_rate"] = 1
            counts["unfinished_task_rate"] = int(not finished)
        if mismatch and finished is not None:
            denominators["interface_recovery_rate"] = 1
            counts["interface_recovery_rate"] = int(
                finished and final_request is not None and final_request > last_error_request
            )
    provider_errors = sum("error_type" in r for r in requests)
    return {
        "sample_id": str(record.get("sample_id", "")),
        "counts": dict(counts),
        "denominators": dict(denominators),
        "model_request_count": len(requests) if observed else None,
        "model_response_count": responses,
        "tool_call_attempt_count": calls_total,
        "tool_return_count": tool_returns,
        "provider_error_count": provider_errors if observed else None,
        "transport_repaired_response_count": sum(
            bool(r.get("transport_repaired")) for r in requests
        ),
        "rule_only": bool(state.get("rule_only")),
        "model_task": bool(requests) or bool(record.get("llm_called")),
        "observed": observed,
        "complete_interface_checks": full_checks and bool(responses),
        "mismatch_turn_indexes": sorted(mismatch_turns["all"]),
        "mismatch_turn_categories": {
            key: sorted(value)
            for key, value in mismatch_turns.items()
            if key != "all"
        },
        "coverage": {
            "tool_names": [name_checks, calls_total],
            "argument_schemas": [argument_checks, calls_total],
            "tool_returns": [return_checks, tool_returns],
        },
        "confirmed_mismatch": mismatch,
        "finished": finished,
        "termination": trace.get("termination") or state.get("termination") or "unknown",
        "tool_return_categories": dict(return_categories),
        "task_correct": contract.task_correct(quality) if quality else None,
        "events": events,
    }


def summarize(rows):
    counts, denominators, coverage, returns, terminations = (Counter() for _ in range(5))
    for row in rows:
        counts.update(row["counts"])
        denominators.update(row["denominators"])
        returns.update(row["tool_return_categories"])
        terminations[row["termination"]] += 1
        for key, (checked, total) in row["coverage"].items():
            coverage[key + "_checked"] += checked
            coverage[key + "_total"] += total
    rates = {name: _rate(counts[name], denominators[name]) for name in RATE_NAMES}
    return {
        "metric_version": METRIC_VERSION,
        "sample_count": len(rows),
        "model_task_count": sum(r["model_task"] for r in rows),
        "rule_only_task_count": sum(r["rule_only"] for r in rows),
        "model_request_count": sum(r["model_request_count"] or 0 for r in rows)
        if all(r["observed"] for r in rows)
        else None,
        "model_response_count": sum(r["model_response_count"] for r in rows),
        "tool_call_attempt_count": sum(r["tool_call_attempt_count"] for r in rows),
        "tool_return_count": sum(r["tool_return_count"] for r in rows),
        "transport_repaired_response_count": sum(
            r["transport_repaired_response_count"] for r in rows
        ),
        "provider_error_task_count": sum(bool(r["provider_error_count"]) for r in rows)
        if all(r["observed"] for r in rows)
        else None,
        "confirmed_mismatch_task_count": sum(r["confirmed_mismatch"] for r in rows),
        "confirmed_mismatch_turn_count": sum(
            len(r.get("mismatch_turn_indexes") or ()) for r in rows
        ),
        "coverage": {
            "raw_trace_tasks": _rate(sum(r["observed"] for r in rows), len(rows)),
            "complete_interface_tasks": _rate(
                denominators["interface_mismatch_task_rate"],
                sum(bool(r["model_response_count"]) for r in rows),
            ),
            "complete_interface_turns": _rate(
                denominators["interface_mismatch_turn_rate"],
                sum(r["model_response_count"] for r in rows),
            ),
            **{
                key: _rate(coverage[key + "_checked"], coverage[key + "_total"])
                for key in ("tool_names", "argument_schemas", "tool_returns")
            },
        },
        "rates": rates,
        "tool_return_categories": dict(returns),
        "terminations": dict(terminations),
        "recovered_task_quality": dict(
            Counter(
                "unknown"
                if r["task_correct"] is None
                else "correct"
                if r["task_correct"]
                else "incorrect"
                for r in rows
                if r["counts"].get("interface_recovery_rate")
            )
        ),
    }


def write_report(rollouts, contract, *, identity, per_sample=()):
    """Recompute from cached rollouts too; never require a new validation run."""
    from src.utils.persistence import atomic_json, atomic_jsonl

    rollouts = Path(rollouts)
    quality = {str(r.get("sample_id")): r for r in per_sample}
    rows = [
        analyze_record(r, contract, quality.get(str(r.get("sample_id"))))
        for r in read_records(rollouts)
    ]
    rows.sort(key=lambda row: row["sample_id"])
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate samples in mismatch report")
    summary = {**summarize(rows), "identity": identity}
    report = rollouts.parent / "mismatch.json"
    samples = rollouts.parent / "mismatch_samples.jsonl"
    atomic_jsonl(samples, rows, sort_keys=True)
    atomic_json(report, summary)
    return {**summary, "report_path": str(report), "samples_path": str(samples)}


def _pair_comparison_view(pair):
    mismatch = pair.get("diagnostics", {}).get("mismatch") or {}
    imr = mismatch.get("rates", {}).get("interface_mismatch_turn_rate") or {}
    coverage = mismatch.get("coverage", {}).get("complete_interface_turns") or {}
    identity = mismatch.get("identity") or {}
    tasks = identity.get("tasks") or {}
    return {
        "pair_id": pair.get("id"),
        "harness": pair.get("harness"),
        "model_id": (pair.get("model") or {}).get("id"),
        "phase": pair.get("phase"),
        "scores": pair.get("scores") or {},
        "imr": imr.get("rate"),
        "imr_count": imr.get("count"),
        "imr_denominator": imr.get("denominator"),
        "imr_observation_coverage": coverage.get("rate"),
        "task_set_sha256": tasks.get("sha256"),
    }


def _selected_vs_latest_trained(pairs, training, selected_pair_id):
    selected = next((pair for pair in pairs if pair.get("id") == selected_pair_id), None)
    completed_training = [item for item in training.values() if item.get("post")]
    latest = completed_training[-1]["post"] if completed_training else None
    if selected is None or latest is None:
        return None
    selected_view = _pair_comparison_view(selected)
    latest_view = _pair_comparison_view(latest)
    same_tasks = (
        bool(selected_view["task_set_sha256"])
        and selected_view["task_set_sha256"] == latest_view["task_set_sha256"]
    )
    same_harness = selected_view["harness"] == latest_view["harness"]
    full_coverage = (
        selected_view["imr_observation_coverage"] == 1
        and latest_view["imr_observation_coverage"] == 1
    )
    comparable = same_tasks and same_harness and full_coverage
    score_names = sorted(set(selected_view["scores"]) | set(latest_view["scores"]))
    score_deltas = {
        name: latest_view["scores"][name] - selected_view["scores"][name]
        for name in score_names
        if isinstance(selected_view["scores"].get(name), (int, float))
        and isinstance(latest_view["scores"].get(name), (int, float))
    }
    selected_imr = selected_view["imr"]
    latest_imr = latest_view["imr"]
    return {
        "comparison": "latest_trained_minus_selected",
        "selected": selected_view,
        "latest_trained": latest_view,
        "selected_model_is_latest_trained": (
            selected_view["model_id"] == latest_view["model_id"]
        ),
        "same_harness": same_harness,
        "same_task_set": same_tasks,
        "full_imr_observation_coverage": full_coverage,
        "comparable": comparable,
        "imr_delta_latest_minus_selected": (
            latest_imr - selected_imr
            if comparable and latest_imr is not None and selected_imr is not None
            else None
        ),
        "quality_deltas_latest_minus_selected": score_deltas if comparable else {},
    }


def round_comparisons(journal, selected_pair_id=None):
    """Expose the already evaluated grid and within-training deltas only."""
    pairs = [p for p in journal.get("evaluations", {}).values() if p.get("split") == "val"]
    entries = [
        {
            "pair_id": p["id"],
            "harness": p["harness"],
            "model_id": p["model"]["id"],
            "scores": p["scores"],
            "mismatch": p.get("diagnostics", {}).get("mismatch"),
        }
        for p in sorted(pairs, key=lambda p: p["id"])
    ]
    changes = []
    for training in journal.get("training", {}).values():
        pre = next((p for p in pairs if p["id"] == training.get("sampling_pair_id")), None)
        post = training.get("post")
        if not pre or not post:
            continue
        before = pre.get("diagnostics", {}).get("mismatch", {})
        after = post.get("diagnostics", {}).get("mismatch", {})
        comparable = (
            before.get("metric_version") == after.get("metric_version") == METRIC_VERSION
            and before.get("coverage", {}).get("complete_interface_turns", {}).get("rate") == 1
            and after.get("coverage", {}).get("complete_interface_turns", {}).get("rate") == 1
        )
        deltas = {}
        for name in RATE_NAMES:
            a = before.get("rates", {}).get(name, {}).get("rate")
            b = after.get("rates", {}).get(name, {}).get("rate")
            deltas[name] = b - a if comparable and a is not None and b is not None else None
        changes.append(
            {
                "training_id": training.get("training_id"),
                "pre_pair_id": pre["id"],
                "post_pair_id": post["id"],
                "comparable": comparable,
                "rate_deltas_post_minus_pre": deltas,
                "quality_deltas_post_minus_pre": {
                    key: post["scores"][key] - value
                    for key, value in pre["scores"].items()
                    if key in post["scores"]
                },
            }
        )
    return {
        "metric_version": METRIC_VERSION,
        "pairs": entries,
        "training_changes": changes,
        "selected_vs_latest_trained": (
            _selected_vs_latest_trained(
                pairs, journal.get("training", {}), selected_pair_id
            )
            if selected_pair_id
            else None
        ),
    }
