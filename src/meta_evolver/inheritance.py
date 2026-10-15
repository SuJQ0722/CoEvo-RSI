"""Independent harness/model modes and deterministic historical retrieval."""

from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter

DEFAULTS = {
    "harness": {"mode": "agent", "max_references": 5, "strategy": "failure_similarity"},
    "model": {
        "mode": "agent",
        "predefined": {"top_k": 3, "density": 0.2, "scale": 1.0},
        "agent": {
            "max_actions": 12,
            "max_evaluations": 8,
            "max_merges": 3,
            "max_merge_parents": 4,
            "merge_methods": ["ties", "linear"],
        },
    },
}


def _mode(value, path):
    # Old resolved configs and INHERITANCE_MODE values remain readable.
    aliases = {"fully_agent": "agent", "full_predefined": "predefined"}
    if not isinstance(value, str):
        raise TypeError(f"{path} must be agent or predefined")
    value = aliases.get(value, value)
    if value not in {"agent", "predefined"}:
        raise ValueError(f"{path} must be agent or predefined")
    return value


def inheritance_settings(config):
    """Normalize modes; explicit component modes override the legacy shared default."""
    if "inheritance" not in config:
        return None
    supplied = config["inheritance"]
    if not isinstance(supplied, dict):
        raise TypeError("inheritance must be a mapping")
    defaults = copy.deepcopy(DEFAULTS)
    if "mode" in supplied:
        legacy = _mode(supplied["mode"], "inheritance.mode")
        for component in ("harness", "model"):
            defaults[component]["mode"] = legacy

    def merge(defaults, supplied, path):
        if not isinstance(supplied, dict):
            raise TypeError(f"{path} must be a mapping")
        unknown = supplied.keys() - defaults.keys()
        if unknown:
            raise ValueError(f"unknown {path} settings: {sorted(unknown)}")
        result = copy.deepcopy(defaults)
        for key, value in supplied.items():
            result[key] = (
                merge(defaults[key], value, f"{path}.{key}")
                if isinstance(defaults[key], dict)
                else copy.deepcopy(value)
            )
        return result

    result = merge(
        defaults, {key: value for key, value in supplied.items() if key != "mode"}, "inheritance"
    )
    for component in ("harness", "model"):
        result[component]["mode"] = _mode(
            result[component]["mode"], f"inheritance.{component}.mode"
        )
    harness, model = result["harness"], result["model"]
    if harness["strategy"] not in {"failure_similarity", "recent"}:
        raise ValueError("inheritance.harness.strategy must be failure_similarity or recent")
    integers = [("harness.max_references", harness["max_references"], 0)]
    integers += [("model.predefined.top_k", model["predefined"]["top_k"], 2)]
    integers += [
        (f"model.agent.{key}", model["agent"][key], minimum)
        for key, minimum in (
            ("max_actions", 1),
            ("max_evaluations", 0),
            ("max_merges", 0),
            ("max_merge_parents", 2),
        )
    ]
    for key, value, minimum in integers:
        if type(value) is not int or value < minimum:
            raise ValueError(f"inheritance.{key} must be an integer >= {minimum}")
    for key, value in model["predefined"].items():
        if key == "top_k":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"inheritance.model.predefined.{key} must be numeric")
        if not math.isfinite(value) or value <= 0 or (key == "density" and value > 1):
            raise ValueError(f"invalid inheritance.model.predefined.{key}")
    methods = model["agent"]["merge_methods"]
    if (
        not isinstance(methods, list)
        or not methods
        or any(method not in {"ties", "linear"} for method in methods)
        or len(set(methods)) != len(methods)
    ):
        raise ValueError("agent merge_methods must be a unique nonempty list of ties/linear")
    return result


def _terms(value):
    # Latin identifiers plus individual CJK characters; deterministic and dependency-free.
    text = json.dumps(value, ensure_ascii=False, sort_keys=True).casefold()
    return Counter(re.findall(r"[a-z][a-z0-9_]{2,}|[\u3400-\u9fff]", text))


def collect_harnesses(inputs, settings):
    """Rank frozen historical cards without an LLM; never select from current siblings."""
    cards = sorted(inputs["history"], key=lambda card: card["id"])
    if len({card["id"] for card in cards}) != len(cards):
        raise ValueError("duplicate harness IDs in frozen history")
    evidence = inputs.get("validation_evidence", inputs.get("train_evidence", []))
    failures = [
        row
        for row in evidence
        if row.get("outcome", {}).get("error")
        or row.get("outcome", {}).get("finished") is False
        or any(
            row.get("outcome", {}).get("metrics", {}).get(key) == 0
            for key in inputs.get("quality_metrics", ())
        )
    ]
    # Never use raw task questions/gold answers as a retrieval feature.
    query = _terms(
        {
            "contract": inputs.get("domain_contract", ""),
            "failures": [
                {"outcome": row.get("outcome", {}), "diagnostics": row.get("diagnostics", {})}
                for row in failures
            ],
        }
    )
    documents = [_terms(card) for card in cards]
    df = Counter(term for document in documents for term in document)
    idf = {term: math.log((1 + len(cards)) / (1 + count)) + 1 for term, count in df.items()}

    def vector(terms):
        return {
            term: (1 + math.log(count)) * idf[term] for term, count in terms.items() if term in idf
        }

    q = vector(query)
    qnorm = math.sqrt(sum(v * v for v in q.values()))
    scores = {}
    for card, document in zip(cards, documents):
        d = vector(document)
        norm = qnorm * math.sqrt(sum(v * v for v in d.values()))
        scores[card["id"]] = (
            sum(value * q.get(term, 0) for term, value in d.items()) / norm if norm else 0.0
        )

    def recency(card):
        generation = card.get("generation", 0)
        generation = generation if isinstance(generation, (int, float)) else 0
        return generation, str(card.get("created_at") or card["id"]), card["id"]

    strategy = settings["strategy"]
    ranked = sorted(
        cards,
        key=lambda c: (scores[c["id"]] if strategy == "failure_similarity" else 0, *recency(c)),
        reverse=True,
    )
    chosen = ranked[: settings["max_references"]]
    receipt = {
        "strategy": strategy,
        "max_references": settings["max_references"],
        "selected_ids": [card["id"] for card in chosen],
        "ranking": [
            {"harness_id": card["id"], "similarity": scores[card["id"]]} for card in ranked
        ],
    }
    return chosen, receipt
