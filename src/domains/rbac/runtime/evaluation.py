"""Evaluate RBAC rollout JSONL and write the four-metric artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.domains.io import read_records
from src.domains.rbac.evaluator import PROTOCOL, evaluate
from src.utils.persistence import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--harness", required=True)
    args = parser.parse_args()
    result = evaluate(list(read_records(args.predictions)))
    payload = {
        "protocol": PROTOCOL,
        "harness": args.harness,
        "total": result.diagnostics["total"],
        **result.metrics,
        **{
            key: value
            for key, value in result.diagnostics.items()
            if key not in {"protocol", "total"}
        },
        "per_sample": result.per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
