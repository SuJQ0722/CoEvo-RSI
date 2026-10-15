"""Command-line entry point for the CoEvoX meta-evolver."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import CANDIDATE_DIR, harness_spec, load_config
from src.meta_evolver.runner import MetaEvolverError, run_meta_evolver
from src.utils.persistence import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend", choices=("claude", "codex"))
    parser.add_argument(
        "--model",
        help="Override the selected meta-evolver backend model for this run.",
    )
    parser.add_argument("--traces", type=Path)
    parser.add_argument("--instruction")
    parser.add_argument("--candidate-name")
    parser.add_argument("--prior-analysis", type=Path)
    parser.add_argument("--prior-feedback", type=Path)
    parser.add_argument(
        "--seed-candidate",
        type=Path,
        help="Resume from a previously rejected candidate bundle.",
    )
    parser.add_argument("--batch-context", type=Path)
    parser.add_argument("--brief", type=Path)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.model:
        meta = config.get("meta_evolver")
        selected_backend = args.backend or (
            str(meta.get("backend", "codex")) if isinstance(meta, dict) else "codex"
        )
        settings = meta.get(selected_backend) if isinstance(meta, dict) else None
        if not isinstance(settings, dict):
            parser.error(f"missing meta_evolver.{selected_backend} configuration")
        settings["model"] = args.model
    project_root = config_path.parent.parent
    selected_harness = harness_spec(config, project_root=project_root)
    meta = config["meta_evolver"]
    trace_path = args.traces or project_root / str(meta["trace_path"])
    try:
        result = run_meta_evolver(
            config=config,
            project_root=project_root,
            trace_path=trace_path,
            backend=args.backend,
            instruction=args.instruction,
            candidate_name=args.candidate_name,
            prior_analysis_path=args.prior_analysis,
            prior_feedback_path=args.prior_feedback,
            seed_candidate_path=args.seed_candidate,
            dry_run=args.dry_run,
            batch_context_path=args.batch_context,
            brief_path=args.brief,
            publish_registry=args.batch_context is None,
        )
    except MetaEvolverError as exc:
        if args.result_json:
            args.result_json.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(
                args.result_json,
                {
                    "status": "rejected",
                    "error": str(exc),
                    "run_dir": str(exc.run_dir),
                    "final_message": str(exc.run_dir / "workspace/final_message.md"),
                    "failure_feedback": str(exc.feedback_path),
                },
            )
        raise SystemExit(2) from exc
    if args.result_json and not args.dry_run:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.result_json,
            {
                "backend": result.backend,
                "run_dir": str(result.run_dir),
                "parent_harness": result.parent_harness,
                "candidate_harness": result.candidate_harness,
                "final_message": str(result.run_dir / "workspace/final_message.md"),
                "evolution_report": str(
                    selected_harness.root
                    / CANDIDATE_DIR
                    / result.candidate_harness
                    / "evolution_report.md"
                ),
                "smoke_passed": result.smoke_passed,
            },
        )
    print(f"backend={result.backend}")
    print(f"run_dir={result.run_dir}")
    print(f"parent_harness={result.parent_harness}")
    print(f"candidate_harness={result.candidate_harness}")
    print(f"smoke_passed={result.smoke_passed}")
    print("command=" + " ".join(result.command))


if __name__ == "__main__":
    main()
