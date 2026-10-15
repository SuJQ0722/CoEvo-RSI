"""Port of HomeBench/code/eval.py compute_accuracy (including comma semantics).

Upstream's top-level driver hardcodes external files and expects absent output2.
We evaluate the supplied output field; Counter multiplicity and brace extraction
are unchanged. Zero-denominator scores are defined as zero. Never execute code
contained in a generated command string.
"""

import re
from collections import Counter, defaultdict
from pathlib import Path

from src.domains.base import EvaluationResult
from src.domains.io import read_records

from .data import PROTOCOL, digest, load
from .submission import SCORING_VERSION, compact_record, official_prediction

METRIC_NAMES = ("em", "f1", "precision", "recall")


def tokens(text, *, prediction):
    text = text.replace(" ", "").replace("\n", "")
    if prediction:
        text = ",".join(re.findall(r"\{(.*?)\}", text))
    else:
        text = text.replace(chr(39) * 3, "")
    return Counter(x for x in text.split(",") if x)


def judge(generated, reference):
    predicted, gold = tokens(generated, prediction=True), tokens(reference, prediction=False)
    return {
        "passed": predicted == gold,
        "tp": sum((predicted & gold).values()),
        "predicted": sum(predicted.values()),
        "gold": sum(gold.values()),
    }


def aggregate(rows):
    rows = list(rows)
    tp = sum(r["tp"] for r in rows)
    pre = sum(r["predicted"] for r in rows)
    gold = sum(r["gold"] for r in rows)
    precision, recall = tp / pre if pre else 0.0, tp / gold if gold else 0.0
    return {
        "em": sum(r["passed"] for r in rows) / len(rows) if rows else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def evaluate(value, *, data_path=None, samples=None, **kwargs):
    samples = list(samples) if samples is not None else list(load(data_path))
    source = read_records(Path(value)) if isinstance(value, (str, Path)) else value
    records = [compact_record(r) for r in source]
    if not samples:
        raise ValueError("empty HomeBench evaluation")
    by_id = {r["sample_id"]: r for r in records}
    if len(by_id) != len(records) or set(by_id) - {s.sample_id for s in samples}:
        raise ValueError("duplicate or unexpected predictions")
    results, groups = [], defaultdict(list)
    for sample in samples:
        record = by_id.get(sample.sample_id)
        if record and record.get("task_digest") != sample.digest:
            raise ValueError(f"prediction/task mismatch: {sample.sample_id}")
        error = record.get("error") if record else "missing_prediction"
        generated = official_prediction(record) if record else ""
        score = judge(generated, sample.record["target"]["commands"])
        if error or not (record or {}).get("finished"):
            score.update(passed=False, tp=0)
        result = {
            "sample_id": sample.sample_id,
            "task_digest": sample.digest,
            "type": sample.record["task_type"],
            "error": error,
            "finished": bool((record or {}).get("finished")),
            **score,
        }
        results.append(result)
        groups[result["type"]].append(result)
    all_metrics = aggregate(results)
    completed = [row for row in results if not row["error"] and row["finished"]]
    completed_metrics = aggregate(completed)
    return EvaluationResult(
        {
            **all_metrics,
            **{f"{key}_all": value for key, value in all_metrics.items()},
            **{f"{key}_completed": value for key, value in completed_metrics.items()},
        },
        results,
        {
            "protocol": PROTOCOL,
            "scoring_version": SCORING_VERSION,
            "sample_count": len(samples),
            "num_results": len(records),
            "completed_count": len(completed),
            "excluded_from_completed_count": len(results) - len(completed),
            "infrastructure_errors": sum(bool(r["error"]) for r in results),
            "dataset_digest": digest(sorted((s.sample_id, s.digest) for s in samples)),
            "per_type": {k: {"count": len(v), **aggregate(v)} for k, v in groups.items()},
        },
    )
