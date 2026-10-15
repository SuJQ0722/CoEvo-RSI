"""Command construction and checkpoint discovery for the ms-swift backend."""

from __future__ import annotations

import json
import os
import subprocess
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path

from src.training.message_schema import validate_training_record
from src.utils.persistence import append_jsonl, atomic_text


def build_ms_swift_sft_command(
    *,
    python: str,
    data: Path,
    model: str,
    output_dir: Path,
    training: dict,
) -> list[str]:
    if training.get("backend", "ms_swift") != "ms_swift":
        raise ValueError("training.backend must be ms_swift")
    if training.get("tuner_type", "full") != "full":
        raise ValueError("CoEvoX requires training.tuner_type=full")

    submodule = Path(str(training.get("submodule_path", "ms-swift")))
    if not (submodule / "swift" / "cli" / "main.py").is_file():
        raise FileNotFoundError(
            f"ms-swift submodule is missing at {submodule}; run git submodule update --init"
        )

    num_processes = int(training.get("num_processes", 1))
    sequence_parallel_size = int(training.get("sequence_parallel_size", 1))
    if num_processes < 1:
        raise ValueError("training.num_processes must be positive")
    configured_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if configured_devices and configured_devices.lower() != "all":
        visible_devices = [
            device.strip() for device in configured_devices.split(",") if device.strip()
        ]
        if len(visible_devices) < num_processes:
            raise ValueError("training.num_processes exceeds the GPUs in CUDA_VISIBLE_DEVICES")
    else:
        visible_devices = [str(index) for index in range(num_processes)]
    training_devices = ",".join(visible_devices[:num_processes])
    if sequence_parallel_size < 1 or num_processes % sequence_parallel_size:
        raise ValueError(
            "training.sequence_parallel_size must be a positive divisor of num_processes"
        )
    truncation_strategy = str(training.get("truncation_strategy", "delete"))
    if truncation_strategy not in {"delete", "left", "right"}:
        raise ValueError(
            "training.truncation_strategy must be delete, left, or right for ms-swift SFT"
        )
    padding_free = bool(training.get("padding_free", False))
    attention_implementation = training.get("attn_impl")
    if padding_free and attention_implementation not in {
        "flash_attn",
        "flash_attention_2",
        "flash_attention_3",
        "flash_attention_4",
    }:
        raise ValueError("training.padding_free=true requires a FlashAttention attn_impl")

    command = [
        "env",
        f"NPROC_PER_NODE={num_processes}",
        f"CUDA_VISIBLE_DEVICES={training_devices}",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f"CELOSS_PARALLEL_SIZE={int(training.get('celoss_parallel_size', 2048))}",
        f"PYTHONPATH={submodule.as_posix()}",
        python,
        "-m",
        "swift.cli.main",
        "sft",
        "--model",
        model,
        "--model_type",
        str(training.get("model_type", "qwen3_5")),
        "--dataset",
        str(data),
        "--output_dir",
        str(output_dir),
        "--tuner_type",
        "full",
        "--freeze_llm",
        "false",
        "--freeze_vit",
        str(bool(training.get("freeze_vit", True))).lower(),
        "--freeze_aligner",
        str(bool(training.get("freeze_aligner", True))).lower(),
        "--use_hf",
        "true",
        "--split_dataset_ratio",
        "0",
        "--torch_dtype",
        str(training.get("torch_dtype", "bfloat16")),
        "--num_train_epochs",
        str(training.get("epochs", 1)),
        "--learning_rate",
        str(training.get("learning_rate", 2e-6)),
        "--per_device_train_batch_size",
        str(training.get("batch_size", 1)),
        "--gradient_accumulation_steps",
        str(training.get("gradient_accumulation_steps", 1)),
        "--max_length",
        str(training.get("max_length", 32768)),
        "--truncation_strategy",
        truncation_strategy,
        "--loss_scale",
        str(training.get("loss_scale", "default")),
        "--agent_template",
        str(training.get("agent_template", "qwen3_5")),
        "--sequence_parallel_size",
        str(sequence_parallel_size),
        "--padding_free",
        str(padding_free).lower(),
        "--deepspeed",
        str(training.get("deepspeed", "zero3")),
        "--gradient_checkpointing",
        str(bool(training.get("gradient_checkpointing", True))).lower(),
        "--optim",
        str(training.get("optimizer", "adamw_torch")),
        "--lr_scheduler_type",
        str(training.get("lr_scheduler_type", "cosine")),
        "--warmup_ratio",
        str(training.get("warmup_ratio", 0.03)),
        "--dataset_num_proc",
        str(training.get("dataset_num_proc", num_processes)),
        "--dataloader_num_workers",
        str(training.get("dataloader_num_workers", 4)),
        "--save_strategy",
        "epoch",
        "--save_only_model",
        "true",
        "--save_total_limit",
        "1",
        "--add_version",
        "false",
        "--report_to",
        "none",
        "--logging_steps",
        str(training.get("logging_steps", 1)),
        "--use_logits_to_keep",
        "false",
    ]
    if attention_implementation:
        command.extend(["--attn_impl", str(attention_implementation)])
    return command


def latest_ms_swift_checkpoint(output_dir: Path) -> Path:
    """Return the highest numbered complete checkpoint emitted by ms-swift."""
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        raise FileNotFoundError(f"ms-swift produced no checkpoint under {output_dir}")
    return max(checkpoints, key=lambda item: item[0])[1]


def resumable_ms_swift_checkpoint(output_dir: Path) -> Path | None:
    """Return the latest fully saved checkpoint, ignoring partial writes."""
    try:
        checkpoint = latest_ms_swift_checkpoint(output_dir)
        step = int(checkpoint.name.removeprefix("checkpoint-"))
        state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return None
    if int(state.get("global_step", -1)) != step or not (checkpoint / "config.json").is_file():
        return None

    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        shards = set((index.get("weight_map") or {}).values())
        if not shards or any(not (checkpoint / shard).is_file() for shard in shards):
            return None
    elif not (checkpoint / "model.safetensors").is_file():
        return None
    return checkpoint


def prepare_ms_swift_dataset(
    source: Path,
    output: Path,
    *,
    tool_policy: str = "preserve",
    sample_id_prefix_repeats: Mapping[str, int] | None = None,
    message_normalizer: Callable[[list[dict]], list[dict]] | None = None,
    terminal_tool_name: str = "final_answer",
) -> int:
    """Normalize accepted rollouts for SFT.

    Production callers preserve every declared tool. The other policies remain
    only for explicitly converting historical artifacts.

    Trainer columns have a fixed metadata schema. Variable domain
    provenance is retained as JSON text so Arrow cannot infer incompatible
    columns or nested structs from different input chunks.
    """

    if tool_policy not in {"terminal_only", "preserve", "plain"}:
        raise ValueError("tool_policy must be terminal_only, preserve, or plain")
    repeat_rules: list[tuple[str, int]] = []
    for raw_prefix, raw_repeat in (sample_id_prefix_repeats or {}).items():
        prefix = str(raw_prefix)
        repeat = int(raw_repeat)
        if not prefix:
            raise ValueError("training.sample_id_prefix_repeats prefixes must not be empty")
        if repeat < 1:
            raise ValueError("training.sample_id_prefix_repeats values must be positive")
        repeat_rules.append((prefix, repeat))
    # Prefer the most specific rule if configured prefixes overlap.
    repeat_rules.sort(key=lambda item: len(item[0]), reverse=True)

    count = 0
    skipped_invalid = 0
    with source.open(encoding="utf-8") as input_stream, atomic_text(output) as output_stream:
        for line_number, line in enumerate(input_stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            # A prepared dataset can be passed through this converter again.
            # Restore provenance before validation and repeat accounting.
            if "metadata_json" in record:
                metadata = json.loads(record.pop("metadata_json"))
                if not isinstance(metadata, dict):
                    raise ValueError(f"invalid SFT metadata at {source}:{line_number}")
                sampling = {**metadata.get("sampling", {}), **record.get("sampling", {})}
                record = {**metadata, **record, "sampling": sampling}
            if not isinstance(record.get("messages"), list):
                raise ValueError(f"missing messages at {source}:{line_number}")  # noqa: TRY004
            if message_normalizer is not None:
                try:
                    record["messages"] = message_normalizer(record["messages"])
                except ValueError:
                    skipped_invalid += 1
                    continue
            assistant_messages = [
                message for message in record["messages"] if message.get("role") == "assistant"
            ]
            already_folded = bool(assistant_messages) and all(
                "reasoning_content" not in message
                and str(message.get("content") or "").lstrip().startswith("<think>")
                for message in assistant_messages
            )
            validate_training_record(
                record,
                require_tool_calls=True,
                require_reasoning_content=not already_folded,
            )
            # ms-swift's dataset preprocessor intentionally drops API-only
            # reasoning fields. Fold them into the model-native training text
            # first, for every domain using the shared SFT path.
            for message in record["messages"]:
                if message.get("role") != "assistant":
                    continue
                reasoning = message.pop("reasoning_content", message.pop("reasoning", None))
                if reasoning is None:
                    continue
                content = str(message.get("content") or "").strip()
                if not content.startswith("<think>"):
                    reasoning = str(reasoning).strip()
                    message["content"] = f"<think>\n{reasoning}\n</think>" + (
                        f"\n\n{content}" if content else ""
                    )
            declared_tools = record.get("tools") or []
            if tool_policy == "terminal_only":
                final_answer_tools = [
                    tool
                    for tool in declared_tools
                    if (tool.get("function") or {}).get("name") == terminal_tool_name
                ]
                if not final_answer_tools:
                    raise ValueError(
                        f"missing declared terminal tool {terminal_tool_name!r} at {source}:{line_number}"
                    )
                record["tools"] = final_answer_tools
            elif not isinstance(declared_tools, list) or not declared_tools:
                if tool_policy == "preserve":
                    raise ValueError(f"missing tools at {source}:{line_number}")
            if tool_policy == "plain":
                record.pop("tools", None)
            sampling = record.setdefault("sampling", {})
            # Older strict records predate tier metadata. Normalize them so a
            # resumed strict/balanced mixture has one trainer-facing schema.
            sampling.setdefault("quality_tier", "strict")
            sampling.setdefault("curation_reason", "strict_correct")
            sampling.setdefault("training_repeat_factor", 1)
            sampling.setdefault("training_repeat_index", 1)
            sample_id = str(record.get("sample_id") or "")
            repeat = next(
                (value for prefix, value in repeat_rules if sample_id.startswith(prefix)),
                1,
            )
            already_repeated = int(sampling.get("training_repeat_factor", 1))
            if already_repeated != 1:
                if already_repeated != repeat:
                    raise ValueError(
                        f"training repeat mismatch at {source}:{line_number}: "
                        f"prepared={already_repeated} configured={repeat}"
                    )
                # A prepared dataset already contains the physical copies.
                repeat = 1
            for repeat_index in range(1, repeat + 1):
                if repeat > 1:
                    sampling["training_repeat_factor"] = repeat
                    sampling["training_repeat_index"] = repeat_index
                # Optional fields must not create columns only in later Arrow
                # chunks (e.g. domain-specific provenance), nor extend the sampling struct.
                # Keep all original metadata, but outside the trainer's schema.
                prepared = {
                    "sample_id": sample_id,
                    "messages": record["messages"],
                    "sampling": {
                        "quality_tier": str(sampling["quality_tier"]),
                        "curation_reason": str(sampling["curation_reason"]),
                        "training_repeat_factor": int(sampling.get("training_repeat_factor", 1)),
                        "training_repeat_index": int(sampling.get("training_repeat_index", 1)),
                    },
                    "metadata_json": json.dumps(
                        {
                            k: v
                            for k, v in record.items()
                            if k not in {"sample_id", "messages", "tools"}
                        },
                        ensure_ascii=False,
                    ),
                }
                if "tools" in record:
                    prepared["tools"] = record["tools"]
                append_jsonl(output_stream, prepared, flush=False)
                count += 1
        if not count:
            raise ValueError(f"no accepted rollouts found in {source}")
    if skipped_invalid:
        warnings.warn(
            f"skipped {skipped_invalid} accepted rollout(s) with invalid ReAct tool calls",
            stacklevel=2,
        )
    return count


def validate_ms_swift_environment(*, python: str, training: dict) -> None:
    """Fail before rollout work when the pinned backend dependencies are unavailable."""
    submodule = Path(str(training.get("submodule_path", "ms-swift")))
    if not (submodule / "swift" / "cli" / "main.py").is_file():
        raise RuntimeError("ms-swift is not initialized; run git submodule update --init ms-swift")
    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        submodule.as_posix()
        if not current_pythonpath
        else submodule.as_posix() + os.pathsep + current_pythonpath
    )
    required_imports = ["swift", "deepspeed", "qwen_vl_utils", "decord"]
    if training.get("padding_free", False):
        required_imports.append("flash_attn")
    completed = subprocess.run(
        [python, "-c", "; ".join(f"import {name}" for name in required_imports)],
        env=environment,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or "").strip().splitlines()[-1:]
        suffix = f" ({detail[0]})" if detail else ""
        raise RuntimeError(
            "ms-swift training dependencies are unavailable; run "
            "pip install -e '.[train]' && pip install -e ms-swift; padding-free SP also needs "
            "MAX_JOBS=4 FLASH_ATTENTION_FORCE_BUILD=TRUE "
            "FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE python -m pip install --force-reinstall "
            "--no-cache-dir --no-build-isolation --no-deps 'flash-attn==2.8.3'" + suffix
        )
