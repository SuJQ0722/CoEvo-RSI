"""Select harness candidates through the configured domain adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.domains.round_adapter import EvaluationArtifact, get_round_adapter


def select_candidate(
    *, config_path, registry_path, parent_name, child_name, stages, report_path, **options
):
    if not stages:
        raise ValueError("at least one selection stage is required")
    config = load_config(config_path)
    adapter = get_round_adapter(config)
    if len(stages) > 1 and not getattr(adapter, "supports_staged_selection", False):
        raise ValueError("this domain requires a single complete evaluation stage")

    def artifact(path):
        resolved = adapter.external_artifact(path.parent, path.stem)
        return EvaluationArtifact(resolved.root, path, resolved.metrics)

    return adapter.select_candidate(
        config=config,
        config_path=config_path,
        registry_path=registry_path,
        parent=parent_name,
        child=child_name,
        stages=stages,
        report_path=report_path,
        parent_artifact=artifact(stages[0][0]),
        child_artifact=artifact(stages[0][1]),
        **options,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--child", required=True)
    parser.add_argument("--stage", nargs=2, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    registry = args.registry or Path(config["meta_evolver"]["registry_path"])
    payload = select_candidate(
        config_path=args.config.resolve(),
        registry_path=registry.resolve(),
        parent_name=args.parent,
        child_name=args.child,
        stages=[(Path(a).resolve(), Path(b).resolve()) for a, b in args.stage],
        report_path=args.report.resolve(),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
