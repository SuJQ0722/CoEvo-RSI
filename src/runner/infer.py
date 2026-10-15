"""Run a configured domain's evaluation commands without assuming a task schema."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import active_harness, load_config
from src.domains.round_adapter import get_round_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--harness")
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    adapter = get_round_adapter(config)
    root = Path(__file__).resolve().parents[2]
    python = adapter.python_executable(config, root, sys.executable)
    artifact = adapter.external_artifact(args.output_dir.resolve(), "evaluation")
    commands = adapter.evaluation_commands(
        python=python,
        config=args.config.resolve(),
        data=args.data.resolve(),
        split="evaluation",
        model=args.model or config["model"].get("name") or config["model"]["base"],
        base_url=args.base_url or config["model"]["base_url"],
        harness=args.harness or active_harness(config),
        max_turns=args.max_turns or int(config["harness"]["max_turns"]),
        concurrency=args.concurrency or int(config.get("rollout", {}).get("concurrency", 1)),
        artifact=artifact,
        limit=args.limit,
        resume=args.resume,
    )
    for command, _ in commands:
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
