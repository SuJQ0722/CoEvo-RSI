"""Normalize Who&When validation rollouts into MetaAgent evidence."""

from __future__ import annotations

from pathlib import Path

from src.domains.evidence import SCHEMA_VERSION, copy_if_normalized, write_records
from src.domains.io import read_records
from src.domains.who_when.runtime.evaluation import StrictAttributionJudge


def build(source: Path, output: Path) -> Path:
    if copy_if_normalized(source, output, domain="who_when"):
        return output
    judge = StrictAttributionJudge()
    sources = (
        [source]
        if source.is_file()
        else sorted(path for path in source.iterdir() if path.suffix == ".jsonl")
    )
    if not sources:
        raise ValueError(f"Who&When evidence source is empty: {source}")

    def normalized_records():
        for record in (record for path in sources for record in read_records(path)):
            prediction = record.get("prediction")
            decision = judge.judge(record) if isinstance(prediction, dict) else None
            yield {
                "schema_version": SCHEMA_VERSION,
                "domain": "who_when",
                "sample_id": record.get("sample_id"),
                "task": record.get("task"),
                "trajectory": record.get("messages") or record.get("trajectory") or [],
                "outcome": {
                    "finished": prediction is not None,
                    "error": record.get("error", ""),
                    "metrics": {
                        "agent_accuracy": float(bool(decision and decision.agent_correct)),
                        "step_accuracy": float(bool(decision and decision.step_correct)),
                    },
                },
                "diagnostics": {
                    key: record.get(key)
                    for key in (
                        "turns",
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "inspected_step_ids",
                        "repeated_inspections",
                    )
                },
                "domain_payload": {
                    "gold": record.get("gold"),
                    "prediction": prediction,
                    "benchmark_split": record.get("benchmark_split"),
                },
            }

    return write_records(normalized_records(), output)
