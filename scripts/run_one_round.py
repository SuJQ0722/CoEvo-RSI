"""Parse configuration and launch a full LINEAGE round or model selection."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import apply_harness_overrides, harness_spec, load_config, resolve_output_root
from src.domains.round_adapter import get_round_adapter
from src.meta_evolver.services import LocalRuntimeServices
from src.training.ms_swift import validate_ms_swift_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("evolve", "model-selection"), default="evolve")
    parser.add_argument(
        "--selection-harness",
        action="append",
        default=[],
        help="Existing harness for standalone model selection; repeat for up to three harnesses",
    )
    parser.add_argument("--backend", choices=("codex", "claude"), default="codex")
    parser.add_argument("--round-number", type=int, default=1)
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument(
        "--base-model-name",
        default="Qwen3.5-9B",
        help="Stable name of the original base model",
    )
    parser.add_argument("--model-path", help="Local/Hugging Face policy checkpoint to evolve")
    parser.add_argument("--base-url")
    parser.add_argument("--vllm-command", help="Override the managed vLLM executable")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        help="Override vLLM tensor parallelism (for example, 4 for four GPUs)",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        help="Override the number of vLLM data-parallel replicas",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help="Override the fraction of memory reserved by vLLM on each GPU",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        help="Override the maximum vLLM context length",
    )
    parser.add_argument(
        "--training-num-processes",
        type=int,
        help="Override the number of distributed full-SFT worker processes",
    )
    parser.add_argument(
        "--training-max-length",
        type=int,
        help="Override the ms-swift SFT sequence length",
    )
    parser.add_argument(
        "--training-sequence-parallel-size",
        type=int,
        help="Override the ms-swift sequence-parallel group size",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        help="Override the full training split configured by data.splits.train",
    )
    parser.add_argument("--candidate-name")
    parser.add_argument(
        "--policy-val-tolerance",
        type=float,
        default=None,
        help="Maximum allowed post-training regression per core validation metric",
    )
    parser.add_argument("--attempts-per-item", type=int)
    parser.add_argument("--sampling-concurrency", type=int)
    parser.add_argument("--inference-concurrency", type=int)
    parser.add_argument("--sampling-quality-tier", choices=("strict", "balanced"))
    parser.add_argument("--max-harness-attempts", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--rebuild-splits", action="store_true")
    parser.add_argument(
        "--resume-parent-artifacts",
        action="store_true",
        help="Resume frozen LINEAGE state using the configured modular identity checks",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stop-after-round",
        action="store_true",
        help="Health-check the resulting policy, then stop vLLM before returning",
    )
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    selected, _ = bootstrap.parse_known_args()
    if selected.config is not None:
        get_round_adapter(load_config(selected.config)).add_arguments(parser)
    args = parser.parse_args()

    if args.stage == "model-selection" and not 1 <= len(args.selection_harness) <= 3:
        parser.error("model-selection requires one to three --selection-harness names")
    if args.stage == "evolve" and args.selection_harness:
        parser.error("--selection-harness is only supported by the model-selection stage")
    if args.round_number < 1:
        parser.error("--round-number must be positive")
    if args.tensor_parallel_size is not None and args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")
    if args.data_parallel_size is not None and args.data_parallel_size < 1:
        parser.error("--data-parallel-size must be positive")
    if args.gpu_memory_utilization is not None and not (0 < args.gpu_memory_utilization < 1):
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if args.max_model_len is not None and args.max_model_len < 1:
        parser.error("--max-model-len must be positive")
    if args.training_num_processes is not None and args.training_num_processes < 1:
        parser.error("--training-num-processes must be positive")
    if args.training_max_length is not None and args.training_max_length < 1:
        parser.error("--training-max-length must be positive")
    if (
        args.training_sequence_parallel_size is not None
        and args.training_sequence_parallel_size < 1
    ):
        parser.error("--training-sequence-parallel-size must be positive")
    services = LocalRuntimeServices(PROJECT_ROOT, training_python=sys.executable)
    config_path = args.config.resolve()
    config = load_config(config_path)
    selected_harness = harness_spec(config, project_root=PROJECT_ROOT)
    adapter = get_round_adapter(config)
    if selected_harness.kind != adapter.harness_kind:
        parser.error(
            f"domain adapter {adapter.name} requires harness kind "
            f"{adapter.harness_kind}, got {selected_harness.kind}"
        )
    try:
        domain_python = adapter.python_executable(config, PROJECT_ROOT, sys.executable)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    parent = str(config.get("harness", {}).get("active") or "initial_harness")
    effective_parent_config = apply_harness_overrides(config, selected_harness.path_for(parent))
    if args.max_turns is None:
        args.max_turns = int((effective_parent_config.get("harness") or {}).get("max_turns", 10))
    if args.max_turns < 1:
        parser.error("--max-turns must be positive")
    output_root = resolve_output_root(config, PROJECT_ROOT)
    splits = config.get("data", {}).get("splits", {})
    split_values = {
        "train": args.train_data or splits.get("train"),
        "val": splits.get("val"),
        "test": splits.get("test"),
    }
    missing_split_config = [name for name, value in split_values.items() if not value]
    if missing_split_config:
        parser.error(
            f"{adapter.name} adapter requires configured data splits: "
            + ", ".join(missing_split_config)
        )
    train_data = services.project_path(split_values["train"])
    val_data = services.project_path(split_values["val"])
    test_data = services.project_path(split_values["test"])
    model_config = config.get("model", {})
    vllm_settings = dict(config.get("vllm") or {})
    if args.vllm_command:
        vllm_settings["command"] = args.vllm_command
    if args.tensor_parallel_size is not None:
        vllm_settings["tensor_parallel_size"] = args.tensor_parallel_size
    if args.data_parallel_size is not None:
        vllm_settings["data_parallel_size"] = args.data_parallel_size
    if args.gpu_memory_utilization is not None:
        vllm_settings["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_model_len is not None:
        vllm_settings["max_model_len"] = args.max_model_len
    vllm_settings.setdefault("max_model_len", model_config.get("max_model_len", 32768))
    server_host = str(vllm_settings.get("host") or "127.0.0.1")
    server_port = int(vllm_settings.get("port", 8000))
    managed_base_url = f"http://{server_host}:{server_port}/v1"
    base_url = args.base_url or str(model_config.get("base_url") or managed_base_url)
    if base_url.rstrip("/") != managed_base_url:
        parser.error(
            f"managed vLLM uses {managed_base_url}; got incompatible --base-url {base_url}"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = args.candidate_name or f"round_{args.round_number:02d}_{stamp}"
    if args.stage == "model-selection":
        round_dir = output_root / "stages" / "model_selection" / candidate
        for name in args.selection_harness:
            try:
                source = selected_harness.path_for(name)
            except ValueError as exc:
                parser.error(str(exc))
            if not source.is_dir():
                parser.error(f"selection harness does not exist: {source}")
    else:
        round_dir = output_root / "rounds" / candidate
    py = sys.executable
    model_source = args.model_path or model_config.get("base") or args.base_model_name
    training_config = dict(config.get("training") or {})
    if args.training_num_processes is not None:
        training_config["num_processes"] = args.training_num_processes
    if args.training_max_length is not None:
        training_config["max_length"] = args.training_max_length
    if args.training_sequence_parallel_size is not None:
        training_config["sequence_parallel_size"] = args.training_sequence_parallel_size
    meta_config = dict(config.get("meta_evolver") or {})
    selection_config = dict(config.get("selection") or {})
    policy_val_tolerance = (
        args.policy_val_tolerance
        if args.policy_val_tolerance is not None
        else float(selection_config.get("policy_validation_tolerance", 0.0))
    )
    if not 0 <= policy_val_tolerance <= adapter.metric_scale_max:
        parser.error(
            "selection.policy_validation_tolerance must be between 0 and "
            f"{adapter.metric_scale_max:g} for {adapter.name}"
        )
    promotion_split = str(selection_config.get("promotion_split", "validation")).strip().lower()
    if promotion_split not in {"validation", "test"}:
        parser.error("selection.promotion_split must be validation or test")
    if not adapter.supports_lineage:
        parser.error(f"domain {adapter.name} is outside the current LINEAGE implementation scope")
    if promotion_split != "validation":
        parser.error("LINEAGE keeps test sealed; selection.promotion_split must be validation")
    sampling_config = dict(config.get("rejection_sampling") or {})
    rollout_config = dict(config.get("rollout") or {})
    inference_concurrency = (
        args.inference_concurrency
        if args.inference_concurrency is not None
        else int(rollout_config.get("concurrency", 1))
    )
    if inference_concurrency < 1:
        parser.error("--inference-concurrency must be positive")
    attempts_per_item = (
        args.attempts_per_item
        if args.attempts_per_item is not None
        else int(sampling_config.get("attempts_per_item", 3))
    )
    if attempts_per_item < 1:
        parser.error("--attempts-per-item must be positive")
    sampling_concurrency = (
        args.sampling_concurrency
        if args.sampling_concurrency is not None
        else int(sampling_config.get("concurrency", 1))
    )
    if sampling_concurrency < 1:
        parser.error("--sampling-concurrency must be positive")
    sampling_quality_tier = args.sampling_quality_tier or str(
        sampling_config.get("quality_tier", "strict")
    )
    max_harness_attempts = (
        args.max_harness_attempts
        if args.max_harness_attempts is not None
        else int(meta_config.get("max_attempts_per_round", 3))
    )
    if max_harness_attempts < 1:
        parser.error("--max-harness-attempts must be positive")
    meta_generation_concurrency = int(
        meta_config.get("generation_concurrency", max_harness_attempts)
    )
    if meta_generation_concurrency < 1:
        parser.error("meta_evolver.generation_concurrency must be positive")
    proposal_batch_size = int(meta_config.get("proposal_batch_size", 3))
    if proposal_batch_size < 1:
        parser.error("meta_evolver.proposal_batch_size must be positive")
    max_shortlisted_children = int(selection_config.get("max_shortlisted_children", 1))
    if max_shortlisted_children < 1:
        parser.error("selection.max_shortlisted_children must be positive")
    if not args.dry_run and args.stage == "evolve":
        validate_ms_swift_environment(python=py, training=training_config)
    missing_split = not (train_data.is_file() and val_data.is_file() and test_data.is_file())
    if args.rebuild_splits or missing_split:
        preparation = adapter.prepare_data_commands(
            python=domain_python,
            project_root=PROJECT_ROOT,
            paths=dict(config.get("paths") or {}),
            splits=splits,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            **adapter.data_preparation_options(args),
        )
        if not preparation and missing_split:
            parser.error(
                f"{adapter.name} data artifacts are missing and its adapter does not "
                "support automatic corpus reconstruction"
            )
        for command, module in preparation:
            services.run(command, module=module, dry_run=args.dry_run)

    from src.meta_evolver.round_runtime import run_lineage

    config["training"] = training_config
    config["vllm"] = vllm_settings
    config["harness"]["max_turns"] = args.max_turns
    config["rollout"]["concurrency"] = inference_concurrency
    config["rejection_sampling"].update(
        attempts_per_item=attempts_per_item,
        concurrency=sampling_concurrency,
        quality_tier=sampling_quality_tier,
    )
    config["meta_evolver"].update(
        max_attempts_per_round=max_harness_attempts,
        proposal_batch_size=proposal_batch_size,
        generation_concurrency=meta_generation_concurrency,
    )
    config["selection"]["policy_validation_tolerance"] = policy_val_tolerance
    run_lineage(
        args=args,
        config=config,
        config_path=config_path,
        project_root=PROJECT_ROOT,
        output_root=output_root,
        round_dir=round_dir,
        domain_python=domain_python,
        model_source=model_source,
        base_url=base_url,
        vllm_settings=vllm_settings,
        training_config=training_config,
        services=services,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
    )


if __name__ == "__main__":
    main()
