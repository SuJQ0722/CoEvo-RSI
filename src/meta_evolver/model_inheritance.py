"""Fixed top-K TIES selection and a resumable agent-driven eval/merge loop."""

from __future__ import annotations

import copy

from src.domains.selection import quality_score

from .lineage import compare_pairs, digest, model_pair_admission, pair_scores, rank_pairs
from .merging import merge_recipe

MODEL_AGENT_PROMPT = """You choose the model initialization for ONE frozen harness.
Read inputs.json. You control which models to evaluate, which checkpoints to
merge, the merge method/parameters, and when to select a starting model. All
scores and diagnostics come from the trusted evaluator on the SAME FULL val set.
Earlier archive scores are context, not comparable measurements on this harness.
Reason about task quality, interface mismatch, recovery and cost together. The
latest model is not assumed best. Respect the action/evaluation/merge budgets.
Actively use evaluation actions to gather comparable full-val evidence before
selecting; do not select latest merely because it is already evaluated. Evaluate
promising checkpoints or merges unless the observed evidence, remaining choices,
or budget makes another evaluation unlikely to change the decision.
Choose the best admissible observed result by quality-first lexicographic ranking:
when selection.ranking_metric_groups is set, maximize each group's weighted mean
in its declared order (also reported as quality_score). Metric weights come from
selection.ranking_metric_weights and default to 1. Follow the supplied
configuration. Inspect validation split scores to identify complementary
checkpoints: models strong on different validation sources can be promising
merge donors. Evaluate
both on this frozen harness before inferring complementarity, and consider TIES
or weighted linear merges within budget. Subset strengths do not guarantee that
weight merging preserves them: evaluate the merged model and select by the same
configured joint objective. Do not use gold labels to route individual examples.
Without ranking_metric_groups, maximize selection.primary_quality_metric first,
then remaining selection.quality_metrics in order; without an explicit primary,
use quality_metrics order. Tolerances are admission gates, not ranking tie bands.
Only when ALL ranking objectives tie within epsilon, prefer lower comparable
interface_mismatch_turn_rate, then lower token cost. Mismatch cannot compensate
for lower task quality or override domain admission. Treat missing or
partial mismatch coverage as unknown, never as zero. In the selection reason,
cite the observed benchmark scores and mismatch evidence and explain why further
evaluation is not worth its cost.
Return ONE JSON object with a nonempty reason and one action:
{"action":"evaluate", "model_id":"...", "reason":"..."}
{"action":"merge", "model_ids":["...","..."], "method":"ties",
 "density":0.2, "scale":1.0, "reason":"..."}
{"action":"merge", "model_ids":["...","..."], "method":"linear",
 "weights":[0.7,0.3], "reason":"..."}
{"action":"select", "model_id":"...", "reason":"..."}
A merge creates AND evaluates the model, consuming one merge and one evaluation.
TIES uses non-base task vectors relative to the materialized common base, per
tensor magnitude trimming, weighted sign election and sign-consistent averaging.
Optional positive weights are supported for TIES too. Linear means a normalized
weighted average of complete weights. Only the listed methods are implemented.
Use exact available model IDs, never paths, shell commands or invented scores.
Select only an already evaluated, evaluable, admissible model for this harness.
You may choose a single checkpoint and skip merging. Respond to tool errors by
correcting your next action. Do not edit harnesses, training, evaluator, thresholds
or files. Do not request test evaluation or additional task samples. No web or
other workspaces: the complete authorized model inventory is in inputs.json.
"""


def _pair_observation(backend, incumbent, pair, selection):
    mismatch = pair.get("diagnostics", {}).get("mismatch", {})
    scores = pair_scores(pair, selection) if pair.get("evaluable", True) else pair.get("scores", {})
    return {
        "pair_id": pair["id"],
        "model_id": pair["model"]["id"],
        "scores": scores,
        "quality_score": list(quality_score(scores, selection))
        if pair.get("evaluable", True)
        else None,
        "token_cost": pair.get("token_cost"),
        "evaluable": pair.get("evaluable", True),
        "admissible": model_pair_admission(backend, incumbent, pair, selection)[0],
        "mismatch": {
            key: mismatch[key]
            for key in (
                "metric_version",
                "rates",
                "coverage",
                "model_task_count",
                "rule_only_task_count",
                "provider_error_task_count",
                "tool_return_categories",
                "terminations",
            )
            if key in mismatch
        },
    }


def _predefined(backend, journal, child, models, incumbent, selection, settings, state):
    if "pairs" not in state:
        state["pairs"] = [backend.evaluate(child, model) for model in models]
        journal.save()
    pairs = state["pairs"]
    singles = [pair for pair in pairs if pair.get("evaluable", True)]
    if "merge_status" not in state:
        base = getattr(backend, "base", None)
        donors = [
            pair
            for pair in rank_pairs(singles, selection, prefer_mismatch=True)
            if base
            and pair["model"].get("common_base_id") == base["effective_weights_hash"]
            and pair["model"]["effective_weights_hash"] != base["effective_weights_hash"]
        ]
        count = settings["top_k"]
        state.update(
            requested_k=count,
            available_donors=len(donors),
            donor_ids=[pair["model"]["id"] for pair in donors[:count]],
        )
        journal.save()
        if base is None:
            state["merge_status"] = "common_base_unavailable"
        elif len(donors) < count:
            # Do not silently turn the fixed-K experiment into an adaptive-K one.
            state["merge_status"] = "insufficient_donors"
        else:
            mixed = backend.merge_candidates(
                [pair["model"] for pair in donors[:count]],
                method="ties",
                density=settings["density"],
                scale=settings["scale"],
            )
            if mixed:
                pair = backend.evaluate(child, mixed)
                # Re-evaluation after interruption is cache-backed and de-duplicated.
                if pair["id"] not in {p["id"] for p in pairs}:
                    pairs.append(pair)
                state["merge_status"] = (
                    "evaluated" if pair.get("evaluable", True) else "unevaluable"
                )
                state["merged_model_id"] = mixed["id"]
            else:
                state["merge_status"] = "merge_failed"
        journal.save()
    return [pair for pair in pairs if pair.get("evaluable", True)]


def _agent_inputs(backend, state, inventory, incumbent, child, selection, settings):
    actions = state["actions"]
    evaluated = sum(action["request"].get("action") in {"evaluate", "merge"} for action in actions)
    merged = sum(action["request"].get("action") == "merge" for action in actions)
    base = getattr(backend, "base", None)
    return {
        "harness": {"id": child["harness"], "hash": child["harness_hash"]},
        "latest_model_id": backend.latest["id"],
        "base_model_id": base["id"] if base else None,
        "selection": selection,
        "models": [
            {
                key: model[key]
                for key in (
                    "id",
                    "route",
                    "parent_ids",
                    "common_base_id",
                    "merge_recipe",
                    "training_id",
                )
                if key in model
            }
            for model in inventory.values()
        ],
        "observations": [
            _pair_observation(backend, incumbent, pair, selection) for pair in state["pairs"]
        ],
        "history": copy.deepcopy(actions),
        "remaining": {
            "actions": settings["max_actions"] - len(actions),
            "evaluations": max(0, settings["max_evaluations"] - evaluated),
            "merges": max(0, settings["max_merges"] - merged),
        },
        "limits": settings,
    }


def _validate_action(request, inventory, settings):
    if (
        not isinstance(request, dict)
        or not isinstance(request.get("reason"), str)
        or not request["reason"].strip()
    ):
        raise ValueError("model agent must return an action with a reason")
    action = request.get("action")
    keys = {"action", "reason"}
    if action in {"select", "evaluate"}:
        keys.add("model_id")
        if not isinstance(request.get("model_id"), str) or request["model_id"] not in inventory:
            raise ValueError("unknown model_id")
    elif action == "merge":
        keys |= {"model_ids", "method", "weights", "density", "scale"}
        ids = request.get("model_ids")
        if (
            not isinstance(ids, list)
            or not 2 <= len(ids) <= settings["max_merge_parents"]
            or any(not isinstance(model_id, str) or model_id not in inventory for model_id in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("merge needs distinct available model IDs within max_merge_parents")
        if request.get("method") not in settings["merge_methods"]:
            raise ValueError("merge method is not enabled")
        if request["method"] == "linear" and ("density" in request or "scale" in request):
            raise ValueError("density and scale apply only to TIES")
    else:
        raise ValueError("model action must be evaluate, merge or select")
    if set(request) - keys:
        raise ValueError(f"unsupported model action fields: {sorted(set(request) - keys)}")


def _agent(backend, journal, child, models, incumbent, selection, settings, state):
    if "pairs" not in state:
        state.update(pairs=[backend.evaluate(child, backend.latest)], actions=[], created_models=[])
        journal.save()
    inventory = {model["id"]: model for model in models + state["created_models"]}
    if "selected_id" in state:
        return [pair for pair in state["pairs"] if pair["id"] == state["selected_id"]]

    while len(state["actions"]) < settings["max_actions"] or (
        state["actions"] and "result" not in state["actions"][-1]
    ):
        if not state["actions"] or "result" in state["actions"][-1]:
            inputs = _agent_inputs(backend, state, inventory, incumbent, child, selection, settings)
            step = len(state["actions"]) + 1
            # Runtime writes a durable MetaAgent receipt, so a crash before journal.save
            # also reuses the same response on restart.
            request = backend.model_inheritance_action(child, step, inputs, MODEL_AGENT_PROMPT)
            if not isinstance(request, dict):
                request = {"invalid_response": str(request)[:1000]}
            state["actions"].append({"request": request})
            journal.save()
        entry = state["actions"][-1]
        request = entry["request"]
        try:
            _validate_action(request, inventory, settings)
            action = request["action"]
            if action == "select":
                selected = next(
                    (
                        pair
                        for pair in state["pairs"]
                        if pair["model"]["id"] == request["model_id"]
                        and pair.get("evaluable", True)
                        and model_pair_admission(backend, incumbent, pair, selection)[0]
                    ),
                    None,
                )
                if selected is None:
                    raise ValueError("select requires an evaluated admissible pair")
                eligible = [
                    pair
                    for pair in state["pairs"]
                    if pair.get("evaluable", True)
                    and model_pair_admission(backend, incumbent, pair, selection)[0]
                ]
                best = rank_pairs(eligible, selection, prefer_mismatch=True)[0]
                if compare_pairs(selected, best, selection, prefer_mismatch=True) > 0:
                    raise ValueError(
                        "select must honor quality-first metric ordering; comparable "
                        "turn-level mismatch and token cost only break quality ties"
                    )
                state.update(
                    selected_id=selected["id"],
                    selection_reason=request["reason"],
                    selection_source="agent",
                )
                entry["result"] = {"selected_pair_id": selected["id"]}
                journal.save()
                return [selected]

            prior = state["actions"][:-1]
            used = sum(e["request"].get("action") in {"evaluate", "merge"} for e in prior)
            if used >= settings["max_evaluations"]:
                raise ValueError("evaluation budget exhausted; select an observed model")
            if action == "merge":
                used_merges = sum(e["request"].get("action") == "merge" for e in prior)
                if used_merges >= settings["max_merges"]:
                    raise ValueError("merge budget exhausted")
                parents = [inventory[model_id] for model_id in request["model_ids"]]
                options = {
                    key: request[key]
                    for key in ("method", "weights", "density", "scale")
                    if key in request
                }
                merge_recipe(parents, base=getattr(backend, "base", None), **options)
                model = backend.merge_candidates(parents, **options)
                if model is None:
                    raise ValueError("merge failed; inspect merge status and choose another action")
                inventory[model["id"]] = model
                if model["id"] not in {m["id"] for m in state["created_models"]}:
                    state["created_models"].append(model)
                journal.save()
            else:
                model = inventory[request["model_id"]]
            pair = backend.evaluate(child, model)
            if pair["id"] not in {p["id"] for p in state["pairs"]}:
                state["pairs"].append(pair)
            entry["result"] = _pair_observation(backend, incumbent, pair, selection)
        except (ValueError, TypeError) as exc:
            entry["result"] = {"error": str(exc)}
        journal.save()

    eligible = [
        pair
        for pair in state["pairs"]
        if pair.get("evaluable", True)
        and model_pair_admission(backend, incumbent, pair, selection)[0]
    ]
    selected = rank_pairs(eligible, selection, prefer_mismatch=True)[:1]
    state.update(
        selected_id=selected[0]["id"] if selected else None,
        selection_source="budget_fallback",
        selection_reason="agent action budget exhausted",
    )
    journal.save()
    return selected


def inherited_model_pairs(backend, journal, shortlisted, incumbent, selection, settings):
    models = journal.once("models", backend.models)
    searches = journal.data.setdefault("model_inheritance", {})
    pool = []
    for child in shortlisted:
        key = child["harness_hash"]
        identity = digest(
            {"harness": key, "settings": settings, "models": [m["id"] for m in models]}
        )
        state = searches.setdefault(
            key,
            {"identity": identity, "mode": settings["model"]["mode"], "harness": child["harness"]},
        )
        if state["identity"] != identity:
            raise ValueError("model inheritance inputs changed on resume")
        journal.save()
        if settings["model"]["mode"] == "agent":
            pairs = _agent(
                backend,
                journal,
                child,
                models,
                incumbent,
                selection,
                settings["model"]["agent"],
                state,
            )
        else:
            pairs = _predefined(
                backend,
                journal,
                child,
                models,
                incumbent,
                selection,
                settings["model"]["predefined"],
                state,
            )
        pool.extend(pairs)
    return pool
