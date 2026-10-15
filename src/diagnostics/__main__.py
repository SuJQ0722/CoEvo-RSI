"""Backfill mismatch reports from existing rollouts without running a model."""

import argparse
import json
from pathlib import Path

from src.domains.round_adapter import get_round_adapter
from src.meta_evolver.weights import file_hash

from .mismatch import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    args = parser.parse_args()
    report = write_report(
        args.rollouts,
        get_round_adapter(args.domain).mismatch_contract(),
        identity={"source": str(args.rollouts.resolve()), "sha256": file_hash(args.rollouts)},
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
