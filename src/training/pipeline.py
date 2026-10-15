"""Run rejection sampling, then optionally launch ms-swift full SFT."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import active_harness, harness_spec, load_config
from src.domains import get_domain_plugin
from src.domains.base import SamplingRequest
from src.training.ms_swift import (
    build_ms_swift_sft_command,
    latest_ms_swift_checkpoint,
    validate_ms_swift_environment,
)
from src.training.preparation import prepare_training_dataset
from src.training.rejection_sampling import collect_to_directory
from src.training.sampling import read_accepted_jsonl
from src.utils.logging import get_logger

LOGGER = get_logger("training.reject_sampling")


def _log_sampling_attempt(record: dict) -> None:
    score = float(record.get("score", bool(record.get("accepted"))))
    LOGGER.info(
        "items=%s attempts=%s accepted_new=%s accepted_total=%s "
        "sample_id=%s score=%.6f accepted=%s phase=%s reason=%s error_detail=%s",
        record.get("items"),
        record.get("attempts_total"),
        record.get("accepted_new"),
        record.get("accepted_total"),
        record.get("sample_id"),
        score,
        record.get("accepted"),
        record.get("phase"),
        record.get("reason"),
        record.get("error_detail") or "",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument(
        "--accepted-rollouts",
        type=Path,
        help=(
            "Skip task-specific rejection sampling and train an already curated "
            "tool-use JSONL through CoEvoX's existing ms-swift backend."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempts-per-item", type=int)
    parser.add_argument("--accepted-per-item", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--quality-tier", choices=("strict", "balanced"))
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--harness")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--sequence-parallel-size", type=int)
    parser.add_argument("--deepspeed")
    parser.add_argument("--optimizer")
    parser.add_argument("--lr-scheduler-type")
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument(
        "--sampling-only",
        action="store_true",
        help="Write accepted rollouts and stop before launching ms-swift",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing sampling artifacts and skip sample IDs already accepted",
    )
    args = parser.parse_args()
    if args.accepted_rollouts is None and args.data is None:
        parser.error("one of --data or --accepted-rollouts is required")
    if args.accepted_rollouts is not None and args.sampling_only:
        parser.error("--sampling-only cannot be combined with --accepted-rollouts")
    config = load_config(args.config)
    harness_name = args.harness or active_harness(config)
    selected_harness = harness_spec(config, project_root=Path(__file__).resolve().parents[2])
    plugin = get_domain_plugin(config)
    if selected_harness.kind != plugin.harness_kind:
        parser.error("configured harness kind does not match its domain")
    if args.max_turns is None:
        args.max_turns = int(config.get("harness", {}).get("max_turns", 10))
    training = config.get("training", {})
    sampling = dict(config.get("rejection_sampling") or {})
    training = dict(training)
    for name, value in {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "optimizer": args.optimizer,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "sequence_parallel_size": args.sequence_parallel_size,
        "deepspeed": args.deepspeed,
    }.items():
        if value is not None:
            training[name] = value
    if not args.sampling_only:
        validate_ms_swift_environment(python=sys.executable, training=training)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = args.output_dir / "accepted_rollouts.jsonl"
    training_path = args.output_dir / "sft_train.jsonl"
    attempts_per_item = (
        args.attempts_per_item
        if args.attempts_per_item is not None
        else int(sampling.get("attempts_per_item", 3))
    )
    accepted_per_item = (
        args.accepted_per_item
        if args.accepted_per_item is not None
        else int(sampling.get("accepted_per_item", 1))
    )
    concurrency = (
        args.concurrency if args.concurrency is not None else int(sampling.get("concurrency", 1))
    )
    quality_tier = args.quality_tier or str(sampling.get("quality_tier", "strict"))
    if attempts_per_item < 1 or accepted_per_item < 1 or concurrency < 1:
        parser.error("sampling counts and concurrency must be positive")

    if args.max_turns < 1:
        parser.error("--max-turns must be positive")
    data_value = args.data or (config.get("data", {}).get("splits", {}).get("train"))
    if not data_value:
        parser.error("training source curation requires --data or data.splits.train")
    train_data = (
        args.data.resolve()
        if args.data is not None
        else (Path(__file__).resolve().parents[2] / data_value).resolve()
    )
    if args.accepted_rollouts is None:
        existing = (
            read_accepted_jsonl(accepted_path) if args.resume and accepted_path.is_file() else []
        )
        stats = collect_to_directory(
            SamplingRequest(
                config_path=args.config.resolve(),
                data_path=train_data,
                output_dir=args.output_dir.resolve(),
                policy_model=args.policy_model,
                base_url=args.base_url,
                harness_name=harness_name,
                attempts_per_item=attempts_per_item,
                accepted_per_item=accepted_per_item,
                concurrency=concurrency,
                max_turns=args.max_turns,
                quality_tier=quality_tier,
                limit=args.limit,
                resume=args.resume,
                temperature=args.temperature,
                on_attempt=_log_sampling_attempt,
            )
        )
        if not len(existing) + stats.accepted:
            raise RuntimeError("rejection sampling accepted no rollouts; checkpoint not trained")
        if args.sampling_only:
            return
    count, training_source = prepare_training_dataset(
        domain=plugin.name,
        quality_tier=quality_tier,
        accepted_rollouts=args.accepted_rollouts or accepted_path,
        training_dataset=training_path,
        train_data=train_data,
        harness_name=harness_name,
        sampling_config=sampling,
        training_config=training,
        config=config,
        val_data=(Path(__file__).resolve().parents[2] / config["data"]["splits"]["val"]).resolve()
        if (training.get("data_curation") or {}).get("mode") == "agent"
        else None,
        project_root=Path(__file__).resolve().parents[2],
        validation_python=sys.executable,
    )
    LOGGER.info(
        "stage=training.dataset records=%d source=%s output=%s",
        count,
        training_source,
        training_path,
    )
    checkpoint_root = args.output_dir / "checkpoint"
    command = build_ms_swift_sft_command(
        python=sys.executable,
        data=training_path,
        model=args.model_path or args.policy_model,
        output_dir=checkpoint_root,
        training=training,
    )
    LOGGER.info("stage=training.ms_swift command=%s", " ".join(command))
    subprocess.run(command, check=True)
    LOGGER.info("checkpoint=%s", latest_ms_swift_checkpoint(checkpoint_root))


if __name__ == "__main__":
    main()
