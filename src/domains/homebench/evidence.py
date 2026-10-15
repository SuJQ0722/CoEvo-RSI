import json

from src.domains.evidence import SCHEMA_VERSION, copy_if_normalized, write_records
from src.domains.io import read_records


def build(source, output):
    if copy_if_normalized(
        source, output, domain="homebench", forbidden_keys={"target", "reference", "gold_output"}
    ):
        return output
    root = source if source.is_dir() else source.parent
    scores = {r["sample_id"]: r for r in json.loads((root / "metrics.json").read_text())["results"]}
    records = read_records(root / "samples.jsonl" if source.is_dir() else source)
    return write_records(
        (
            {
                "schema_version": SCHEMA_VERSION,
                "domain": "homebench",
                "sample_id": r["sample_id"],
                "task": {"instruction": r["instruction"], "tools": r["tools"]},
                "trajectory": r["messages"],
                "outcome": {
                    "finished": r["finished"],
                    "error": r["error"],
                    "metrics": {"em": float(scores[r["sample_id"]]["passed"])},
                },
                "diagnostics": {"task_type": r["task_type"]},
                "domain_payload": {"protocol": r["protocol"]},
            }
            for r in records
        ),
        output,
    )
