"""Same-task comparison: improve EM, gate F1, reject infrastructure artifacts."""

import math

from src.domains.base import SelectionDecision


def select(parent, child, config):
    keys = ("em", "f1")
    deltas = {
        k: child.metrics.get(k, float("nan")) - parent.metrics.get(k, float("nan")) for k in keys
    }

    def reject(reason):
        return SelectionDecision(False, reason, deltas)

    if any(
        not math.isfinite(result.metrics.get(k, float("nan"))) or not 0 <= result.metrics[k] <= 1
        for result in (parent, child)
        for k in keys
    ):
        return reject("invalid EM/F1")
    for result in (parent, child):
        if result.diagnostics.get("infrastructure_errors", 0):
            return reject("infrastructure errors must be retried before selection")
        if not result.per_sample:
            return reject("missing per-sample identity")
    a = {r["sample_id"]: r["task_digest"] for r in parent.per_sample}
    b = {r["sample_id"]: r["task_digest"] for r in child.per_sample}
    if a != b or len(a) != len(parent.per_sample) or len(b) != len(child.per_sample):
        return reject("parent/child evaluation tasks differ")
    epsilon = float(config.get("epsilon", 1e-12))
    if deltas["f1"] < -float(config.get("f1_tolerance", 0.0)) - epsilon:
        return reject("command F1 regressed")
    if deltas["em"] <= epsilon:
        return reject("no strict EM improvement")
    return SelectionDecision(True, "EM improved with command F1 non-regression", deltas)
