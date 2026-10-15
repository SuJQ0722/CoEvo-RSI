"""Produce score- and routing-aware evidence for RBAC MetaEvolver."""

from __future__ import annotations

from pathlib import Path

from src.domains.evidence import SCHEMA_VERSION, copy_if_normalized, write_records
from src.domains.io import read_records

from .evaluator import evaluate


def build(source: Path, output: Path) -> Path:
    if copy_if_normalized(source, output, domain="rbac"):
        return output
    rollouts = source if source.is_file() else source / "samples.jsonl"
    if not rollouts.is_file():
        raise FileNotFoundError(f"RBAC rollout artifact not found: {rollouts}")
    records = list(read_records(rollouts))
    scored = {str(item["sample_id"]): item for item in evaluate(records).per_sample}

    def normalized_records():
        for record in records:
            sample_id = str(record.get("sample_id") or "")
            score = scored[sample_id]
            prediction = record.get("prediction") or {}
            yield {
                "schema_version": SCHEMA_VERSION,
                "domain": "rbac",
                "sample_id": sample_id,
                "task": {
                    "instruction": record.get("instruction"),
                    "test_role": record.get("test_role"),
                },
                "trajectory": record.get("messages") or [],
                "outcome": {
                    "finished": not bool(record.get("error")),
                    "error": record.get("error", ""),
                    "metrics": {
                        "score": score["score"],
                        "llm_called": score["llm_called"],
                        "rule_used": score["rule_used"],
                        "rule_correct": score["rule_correct"],
                    },
                },
                "diagnostics": {
                    "route_source": prediction.get("route_source"),
                    "rule_confidence": prediction.get("rule_confidence"),
                    "prompt_tokens": record.get("prompt_tokens", 0),
                    "completion_tokens": record.get("completion_tokens", 0),
                },
                "domain_payload": {
                    "test_type": record.get("test_type"),
                    "predicted_access": score["predicted"],
                    "gold_access": score["gold"],
                    "response": prediction.get("response", ""),
                },
            }

    return write_records(normalized_records(), output)

