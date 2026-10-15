"""Shared LINEAGE scheduling and selection, independent of model/CLI backends."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cmp_to_key
from pathlib import Path
from typing import Any

from src.domains.selection import (
    quality_names,
    quality_score,
    ranking_groups,
    split_quality_scores,
)
from src.utils.persistence import atomic_json

LOGGER = logging.getLogger("pipeline.lineage")


def resume_config(config: Any) -> Any:
    """Return a normalized config for provenance and diagnostic hashes.

    Round resume no longer rejects configuration changes. Frozen artifacts and
    pair identities remain authoritative for already completed operations.
    """
    normalized = copy.deepcopy(config)
    if not isinstance(normalized, dict):
        return normalized
    meta = normalized.get("meta_evolver")
    if not isinstance(meta, dict):
        return normalized
    meta.pop("model", None)  # compatibility with flat/custom configurations
    for backend in ("codex", "claude"):
        settings = meta.get(backend)
        if isinstance(settings, dict):
            settings.pop("model", None)
    return normalized


def _log_identity_mismatch(frozen: dict, current: dict, changes: set[str]) -> None:
    """Expose actionable resume differences before rejecting frozen state."""
    if "code_hash" in changes:
        LOGGER.error(
            "resume identity mismatch field=code_hash frozen=%s current=%s",
            frozen.get("code_hash"),
            current.get("code_hash"),
        )
    if "environment_version" not in changes:
        return
    frozen_environment = frozen.get("environment_version") or {}
    current_environment = current.get("environment_version") or {}
    frozen_packages = frozen_environment.get("packages") or {}
    current_packages = current_environment.get("packages") or {}
    added = {
        name: current_packages[name]
        for name in sorted(current_packages.keys() - frozen_packages.keys())
    }
    removed = {
        name: frozen_packages[name]
        for name in sorted(frozen_packages.keys() - current_packages.keys())
    }
    changed = {
        name: {"frozen": frozen_packages[name], "current": current_packages[name]}
        for name in sorted(frozen_packages.keys() & current_packages.keys())
        if frozen_packages[name] != current_packages[name]
    }
    LOGGER.error(
        "resume identity mismatch field=environment_version domain_python frozen=%s current=%s",
        frozen_environment.get("domain_python"),
        current_environment.get("domain_python"),
    )
    LOGGER.error(
        "resume environment packages added=%s",
        json.dumps(added, sort_keys=True, ensure_ascii=False),
    )
    LOGGER.error(
        "resume environment packages removed=%s",
        json.dumps(removed, sort_keys=True, ensure_ascii=False),
    )
    LOGGER.error(
        "resume environment package versions changed=%s",
        json.dumps(changed, sort_keys=True, ensure_ascii=False),
    )


def _identity_field_changed(key: str, frozen: Any, current: Any) -> bool:
    if key == "config":
        # Configuration changes are allowed when resuming an interrupted
        # round. Durable journal entries keep completed operations frozen;
        # other identity fields still protect data and lineage continuity.
        return False
    if key != "environment_version":
        return frozen != current
    # Executable aliases such as /usr/bin/python and /usr/bin/python3 do not
    # change the installed runtime. Package names and versions remain strict.
    frozen_environment = frozen or {}
    current_environment = current or {}
    return (frozen_environment.get("packages") or {}) != (current_environment.get("packages") or {})


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


class Journal:
    """Controller-only, durable inputs and results; callers hold the round lock."""

    def __init__(
        self,
        path: Path,
        identity: dict,
        *,
        allow_code_change: bool = False,
        ignored_identity_fields: set[str] | None = None,
    ):
        self.path = path
        self.data = (
            json.loads(path.read_text())
            if path.exists()
            else {
                "identity": identity,
                "batches": [],
                "evaluations": {},
                "training": {},
            }
        )
        frozen_identity = self.data["identity"]
        ignored_identity_fields = ignored_identity_fields or set()
        identity_changes = {
            key
            for key in frozen_identity.keys() | identity.keys()
            if key not in ignored_identity_fields
            if _identity_field_changed(key, frozen_identity.get(key), identity.get(key))
        }
        if identity_changes and not (allow_code_change and identity_changes == {"code_hash"}):
            _log_identity_mismatch(frozen_identity, identity, identity_changes)
            raise ValueError(
                "round inputs changed; resume requires the frozen configuration and data "
                f"(changed top-level fields: {', '.join(sorted(identity_changes))})"
            )
        if identity_changes:
            LOGGER.warning(
                "resume accepts controller code revision frozen=%s current=%s",
                frozen_identity["code_hash"],
                identity["code_hash"],
            )
            revisions = self.data.setdefault("resume_code_hashes", [])
            if identity["code_hash"] not in revisions:
                revisions.append(identity["code_hash"])
        self.save()

    def save(self):
        atomic_json(self.path, self.data)

    def once(self, key, operation):
        if key not in self.data:
            self.data[key] = operation()
            self.save()
        return self.data[key]


def candidate_name_for_slot(round_id: str, slot: int) -> str:
    """Stable names for reserved editor slots, including interrupted proposals."""
    return round_id if slot == 1 else f"{round_id}_retry_{slot:02d}"


def pair_scores(pair, selection):
    """Read missing split metrics from old artifacts, without changing journal history."""
    scores = dict(pair["scores"])
    required = {name for group in ranking_groups(selection) for name in group}
    if required - scores.keys():
        path = (pair.get("artifact") or {}).get("metrics")
        if path:
            payload = json.loads(Path(path).read_text())
            restored = split_quality_scores(payload, quality_names(selection))
            scores = {**restored, **scores}
    return scores


def tolerance(selection, name):
    return float(
        selection.get(
            f"{name}_tolerance",
            (selection.get("tolerances") or {}).get(name, selection.get("quality_tolerance", 0.0)),
        )
    )


def _mismatch_value(pair, selection):
    metric_name = selection.get("mismatch_metric")
    if not metric_name:
        return None
    report = pair.get("diagnostics", {}).get("mismatch") or {}
    rate = (report.get("rates", {}).get(metric_name) or {}).get("rate")
    coverage_name = selection.get("mismatch_coverage_metric", "complete_interface_turns")
    coverage = (report.get("coverage", {}).get(coverage_name) or {}).get("rate")
    tasks = (report.get("identity") or {}).get("tasks") or {}
    task_set = tasks.get("sha256")
    if (
        not isinstance(rate, (int, float))
        or not math.isfinite(float(rate))
        or not isinstance(coverage, (int, float))
        or float(coverage) < 1.0
        or not task_set
    ):
        return None
    return float(rate), str(task_set), report.get("metric_version")


def _compare_mismatch(a, b, selection):
    """Compare fully observed turn-level mismatch on the same task set."""

    left, right = _mismatch_value(a, selection), _mismatch_value(b, selection)
    if left is None or right is None or left[1:] != right[1:]:
        return 0
    epsilon = float(selection.get("mismatch_improvement_epsilon", selection.get("epsilon", 1e-12)))
    if abs(left[0] - right[0]) <= epsilon:
        return 0
    return -1 if left[0] < right[0] else 1


def model_pair_admission(backend, incumbent, pair, selection):
    """Respect domain quality gates; mismatch cannot override a rejection."""
    return backend.admit(incumbent, pair)


def compare_pairs(a, b, selection, *, final=False, prefer_mismatch=False):
    """Negative means a wins. Tolerances never collapse distinct quality scores."""
    epsilon = float(selection.get("epsilon", 1e-12))
    # Cached vectors may reflect an older objective. Reconstruct from raw metrics.
    avs = quality_score(pair_scores(a, selection), selection)
    bvs = quality_score(pair_scores(b, selection), selection)
    for av, bv in zip(avs, bvs):
        av, bv = float(av), float(bv)
        if not math.isfinite(av) or not math.isfinite(bv):
            raise ValueError("non-finite quality score")
        if abs(av - bv) > epsilon:
            return -1 if av > bv else 1
    if prefer_mismatch:
        mismatch_order = _compare_mismatch(a, b, selection)
        if mismatch_order:
            return mismatch_order
    # Unknown usage is never treated as free. JSON stores it as null.
    ac = a.get("token_cost")
    bc = b.get("token_cost")
    ac = math.inf if ac is None else float(ac)
    bc = math.inf if bc is None else float(bc)
    if ac != bc:
        return -1 if ac < bc else 1
    if final:
        ak = (a.get("phase", "pre") != "pre", a.get("candidate_rank", 0), a["id"])
        bk = (b.get("phase", "pre") != "pre", b.get("candidate_rank", 0), b["id"])
    else:
        ak, bk = a["id"], b["id"]
    return (ak > bk) - (ak < bk)


def rank_pairs(pairs, selection, *, final=False, prefer_mismatch=False):
    unique = {}
    for pair in pairs:
        key = (pair["harness_hash"], pair["model"]["effective_weights_hash"])
        previous = unique.get(key)
        if (
            previous is None
            or compare_pairs(
                pair,
                previous,
                selection,
                final=final,
                prefer_mismatch=prefer_mismatch,
            )
            < 0
        ):
            unique[key] = pair
    return sorted(
        unique.values(),
        key=cmp_to_key(
            lambda a, b: compare_pairs(
                a,
                b,
                selection,
                final=final,
                prefer_mismatch=prefer_mismatch,
            )
        ),
    )


def shortlist_harnesses(pairs, selection):
    """Keep up to two leaders; admit a third only when it is close to the second."""
    ranked = rank_pairs(pairs, selection)[: int(selection["max_shortlisted_children"])]
    if len(ranked) == 3:
        second, third = ranked[1:]
        epsilon = float(selection.get("epsilon", 1e-12))
        if any(
            abs(second["scores"][name] - third["scores"][name])
            > tolerance(selection, name) + epsilon
            for name in quality_names(selection)
        ):
            return ranked[:2]
    return ranked


def fallback_decision(first, second, post, selection):
    training_limit = int(selection.get("max_training_candidates", 2))
    if training_limit not in {1, 2}:
        raise ValueError("selection.max_training_candidates must be 1 or 2")
    epsilon = float(selection.get("epsilon", 1e-12))
    gaps = (
        {
            name: abs(first["scores"][name] - second["scores"][name])
            for name in quality_names(selection)
        }
        if second
        else {}
    )
    drops = (
        {name: first["scores"][name] - post["scores"][name] for name in quality_names(selection)}
        if post
        else {}
    )
    close = bool(second) and all(
        gap <= tolerance(selection, name) + epsilon for name, gap in gaps.items()
    )
    regressed = bool(post) and any(
        drop > float(selection["policy_validation_tolerance"]) + epsilon for drop in drops.values()
    )
    return {
        "close": close,
        "regressed": regressed,
        "triggered": training_limit > 1 and close and regressed,
        "gaps": gaps,
        "drops": drops,
        "reason": (
            "training_candidate_limit"
            if training_limit == 1
            else "fallback"
            if close and regressed
            else "no_second_candidate"
            if not second
            else "second_candidate_too_far"
            if not close
            else "no_evaluable_post"
            if not post
            else "first_post_within_tolerance"
        ),
    }


def select_model_pairs(
    backend, journal: Journal, shortlisted: list, incumbent: dict, selection: dict
):
    """Run configured model inheritance on full val; cache the top two pairs.

    Shared by evolution and the standalone model-selection stage. Does not
    generate harnesses, train models, or publish an incumbent.
    """
    if "ranking" not in journal.data:
        from .inheritance import inheritance_settings
        from .model_inheritance import inherited_model_pairs

        settings = inheritance_settings(getattr(backend, "config", {}))
        if settings:
            pool = inherited_model_pairs(
                backend, journal, shortlisted, incumbent, selection, settings
            )
        else:
            # Compatibility for frozen/custom configurations without the new section.
            models = journal.once("models", backend.models)
            pool = []
            for child in shortlisted:
                singles = [backend.evaluate(child, model) for model in models]
                singles = [pair for pair in singles if pair.get("evaluable", True)]
                if not singles:
                    continue
                best = rank_pairs(singles, selection)[0]
                mixed = backend.merge(best["model"], backend.latest)
                if mixed:
                    singles.append(backend.evaluate(child, mixed))
                pool.extend(pair for pair in singles if pair.get("evaluable", True))
        eligible = [
            pair for pair in pool if model_pair_admission(backend, incumbent, pair, selection)[0]
        ]
        journal.data["pair_pool"] = pool
        journal.data["ranking"] = [
            dict(pair, candidate_rank=index, phase="pre")
            for index, pair in enumerate(
                rank_pairs(eligible, selection, prefer_mismatch=True)[:2], 1
            )
        ]
        journal.save()
    return journal.data["ranking"]


def evolve_round(backend, journal: Journal, config: dict) -> dict:
    """Run the configured proposal batch, traversal and training loops.

    Backend operations persist their own expensive sub-stages before returning,
    so a lost controller process can resume the same logical operation.
    """
    if "outcome" in journal.data:
        return backend.finalize(journal.data["outcome"], journal)
    meta, selection = config["meta_evolver"], config["selection"]
    if int(selection.get("max_training_candidates", 2)) not in {1, 2}:
        raise ValueError("selection.max_training_candidates must be 1 or 2")
    width, budget = int(meta["proposal_batch_size"]), int(meta["max_attempts_per_round"])
    workers = int(meta["generation_concurrency"])
    limit = int(selection["max_shortlisted_children"])
    if min(width, budget, workers, limit) < 1 or limit > 3:
        raise ValueError(
            "positive batch/worker/budget values and a harness shortlist of 1–3 are required"
        )
    incumbent_cached = "incumbent_eval" in journal.data
    LOGGER.info("lineage incumbent evaluation start cached=%s", incumbent_cached)
    incumbent = journal.once("incumbent_eval", backend.incumbent_eval)
    LOGGER.info(
        "lineage incumbent evaluation complete cached=%s harness=%s",
        incumbent_cached,
        incumbent.get("harness", incumbent.get("id", "unknown")),
    )
    context_cached = "initial_context" in journal.data
    LOGGER.info("meta curator start cached=%s", context_cached)
    context = journal.once("initial_context", backend.curate)
    LOGGER.info(
        "meta curator complete cached=%s references=%d context=%s",
        context_cached,
        len(context.get("references") or []),
        context.get("id", "unknown"),
    )
    feedback = []
    seen = {incumbent["harness_hash"]}
    shortlisted = []
    used = 0
    batch_index = 0
    while used < budget:
        count = min(width, budget - used)
        batches = journal.data["batches"]
        if batch_index == len(batches):
            batch_id = f"{backend.round_id}_b{batch_index + 1:02d}"
            batches.append(
                {
                    "id": batch_id,
                    "count": count,
                    "slots": [used + i + 1 for i in range(count)],
                    "prior_failure_summary_ids": [item["id"] for item in feedback],
                }
            )
            journal.save()  # reserve every slot before any editor starts
        batch = batches[batch_index]
        if "briefs" not in batch:
            LOGGER.info(
                "meta briefs start batch=%s count=%d prior_failure_summaries=%d",
                batch["id"],
                count,
                len(feedback),
            )
            batch["context"], batch["briefs"] = backend.briefs(context, feedback, batch)
            if len(batch["briefs"]) != count:
                raise ValueError("MetaAgent must produce exactly the current batch's brief count")
            journal.save()
            LOGGER.info("meta briefs complete batch=%s count=%d", batch["id"], count)
        else:
            LOGGER.info("meta briefs reuse batch=%s count=%d", batch["id"], len(batch["briefs"]))
        if "proposals" not in batch:
            active_workers = min(workers, count)
            LOGGER.info(
                "meta editors dispatch batch=%s count=%d workers=%d slots=%s",
                batch["id"],
                count,
                active_workers,
                ",".join(str(slot) for slot in batch["slots"]),
            )

            def edit_one(args, context=batch["context"], batch=batch):
                brief, slot = args
                candidate = candidate_name_for_slot(backend.round_id, slot)
                started = time.monotonic()
                LOGGER.info(
                    "meta editor start batch=%s slot=%d candidate=%s brief=%s",
                    batch["id"],
                    slot,
                    candidate,
                    brief.get("id", "unknown"),
                )
                try:
                    proposal = backend.edit(context, brief, slot)
                except Exception:
                    LOGGER.exception(
                        "meta editor failed batch=%s slot=%d candidate=%s wall_seconds=%.3f",
                        batch["id"],
                        slot,
                        candidate,
                        time.monotonic() - started,
                    )
                    raise
                LOGGER.info(
                    "meta editor complete batch=%s slot=%d candidate=%s valid=%s "
                    "wall_seconds=%.3f reason=%s",
                    batch["id"],
                    slot,
                    candidate,
                    proposal.get("valid"),
                    time.monotonic() - started,
                    proposal.get("reason", ""),
                )
                return proposal

            with ThreadPoolExecutor(max_workers=min(workers, count)) as pool:
                batch["proposals"] = list(pool.map(edit_one, zip(batch["briefs"], batch["slots"])))
            journal.save()
            LOGGER.info(
                "meta editors complete batch=%s count=%d valid=%d",
                batch["id"],
                len(batch["proposals"]),
                sum(bool(item.get("valid")) for item in batch["proposals"]),
            )
        else:
            LOGGER.info(
                "meta editors reuse batch=%s count=%d",
                batch["id"],
                len(batch["proposals"]),
            )
        if "results" not in batch:
            batch["results"] = []
        # All editors finish before the controller evaluates any sibling.
        for index, proposal in enumerate(batch["proposals"]):
            if index < len(batch["results"]):
                stored = batch["results"][index]
                if stored.get("harness_hash"):
                    seen.add(stored["harness_hash"])
                continue
            hhash = proposal.get("harness_hash")
            if not proposal.get("valid") or not hhash or hhash in seen:
                result = {
                    "proposal": proposal,
                    "accepted": False,
                    "reason": "duplicate_harness"
                    if hhash in seen
                    else proposal.get("reason", "invalid_child"),
                }
            else:
                seen.add(hhash)
                pair = backend.evaluate(proposal, backend.latest)
                accepted, reason = backend.admit(incumbent, pair)
                result = {
                    "proposal": proposal,
                    "pair": pair,
                    "harness_hash": hhash,
                    "accepted": accepted,
                    "reason": reason,
                }
            batch["results"].append(result)
            journal.save()
        used += count
        passed = [result["pair"] for result in batch["results"] if result["accepted"]]
        if passed:
            shortlisted = shortlist_harnesses(passed, selection)
            break
        if used < budget:
            if "attribution" not in batch:
                batch["attribution"] = backend.summarize(batch)
                journal.save()
            feedback.append(batch["attribution"])
        batch_index += 1
    journal.data["shortlisted"] = shortlisted
    journal.save()
    if not shortlisted:
        outcome = {"status": "no_update", "reason": "no_passing_child", "winner": incumbent}
    else:
        ranking = select_model_pairs(backend, journal, shortlisted, incumbent, selection)
        if not ranking:
            outcome = {"status": "no_update", "reason": "no_admissible_pair", "winner": incumbent}
        else:
            first, second = ranking[0], ranking[1] if len(ranking) > 1 else None

            def train(pair):
                key = str(pair["candidate_rank"])
                if key not in journal.data["training"]:
                    journal.data["training"][key] = backend.train(pair)
                    journal.save()
                run = journal.data["training"][key]
                return run.get("post")

            post = train(first)
            decision = fallback_decision(first, second, post, selection)
            journal.data["fallback"] = decision
            journal.save()
            available = [first] + ([post] if post else [])
            if decision["triggered"]:
                second_post = train(second)
                available += [second] + ([second_post] if second_post else [])
            selected = rank_pairs(available, selection, final=True, prefer_mismatch=True)[0]
            accepted, reason = model_pair_admission(backend, incumbent, selected, selection)
            outcome = {
                "status": ("trained_pair" if selected["phase"] == "post" else "untrained_pair")
                if accepted
                else "no_update",
                "reason": reason,
                "selected": selected,
                "winner": selected if accepted else incumbent,
            }
    journal.data["outcome"] = outcome
    journal.save()
    return backend.finalize(outcome, journal)
