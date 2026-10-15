import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.training.ms_swift import (
    build_ms_swift_sft_command,
    latest_ms_swift_checkpoint,
    prepare_ms_swift_dataset,
    resumable_ms_swift_checkpoint,
)


def test_ms_swift_command_uses_full_zero3_and_sequence_parallelism(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    command = build_ms_swift_sft_command(
        python="python",
        data=Path("data.jsonl"),
        model="Qwen/Qwen3.5-4B",
        output_dir=Path("checkpoint"),
        training={
            "backend": "ms_swift",
            "submodule_path": "ms-swift",
            "tuner_type": "full",
            "num_processes": 8,
            "sequence_parallel_size": 8,
            "padding_free": True,
            "attn_impl": "flash_attn",
            "deepspeed": "zero3",
        },
    )

    assert command[:7] == [
        "env",
        "NPROC_PER_NODE=8",
        "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "CELOSS_PARALLEL_SIZE=2048",
        "PYTHONPATH=ms-swift",
        "python",
    ]
    assert command[command.index("--tuner_type") + 1] == "full"
    assert command[command.index("--freeze_llm") + 1] == "false"
    assert command[command.index("--deepspeed") + 1] == "zero3"
    assert command[command.index("--sequence_parallel_size") + 1] == "8"
    assert command[command.index("--padding_free") + 1] == "true"
    assert command[command.index("--attn_impl") + 1] == "flash_attn"
    assert command[command.index("--truncation_strategy") + 1] == "delete"
    assert command[command.index("--agent_template") + 1] == "qwen3_5"
    assert command[command.index("--model_type") + 1] == "qwen3_5"


def test_ms_swift_command_limits_visible_gpus_to_process_count(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    command = build_ms_swift_sft_command(
        python="python",
        data=Path("data.jsonl"),
        model="model",
        output_dir=Path("checkpoint"),
        training={
            "submodule_path": "ms-swift",
            "num_processes": 2,
        },
    )

    assert "NPROC_PER_NODE=2" in command
    assert "CUDA_VISIBLE_DEVICES=0,1" in command
    assert command[command.index("--tuner_type") + 1] == "full"


def test_ms_swift_command_rejects_non_full_training():
    with pytest.raises(ValueError, match="training.tuner_type=full"):
        build_ms_swift_sft_command(
            python="python",
            data=Path("data.jsonl"),
            model="model",
            output_dir=Path("checkpoint"),
            training={"tuner_type": "unsupported"},
        )


def test_sequence_parallel_size_must_divide_world_size():
    with pytest.raises(ValueError, match="positive divisor"):
        build_ms_swift_sft_command(
            python="python",
            data=Path("data.jsonl"),
            model="model",
            output_dir=Path("checkpoint"),
            training={
                "submodule_path": "ms-swift",
                "num_processes": 6,
                "sequence_parallel_size": 4,
            },
        )


def test_split_truncation_is_rejected_for_sft():
    with pytest.raises(ValueError, match="delete, left, or right"):
        build_ms_swift_sft_command(
            python="python",
            data=Path("data.jsonl"),
            model="model",
            output_dir=Path("checkpoint"),
            training={
                "submodule_path": "ms-swift",
                "truncation_strategy": "split",
            },
        )


def test_latest_ms_swift_checkpoint_uses_largest_step(tmp_path):
    (tmp_path / "checkpoint-2").mkdir()
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-invalid").mkdir()

    assert latest_ms_swift_checkpoint(tmp_path) == tmp_path / "checkpoint-10"


def test_prepare_ms_swift_dataset_rejects_noncanonical_input(tmp_path):
    data = tmp_path / "accepted.jsonl"
    data.write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "answer"}]}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reasoning_content"):
        prepare_ms_swift_dataset(data, data)


def test_prepare_ms_swift_dataset_repeats_matching_sample_prefix(tmp_path):
    source = tmp_path / "accepted.jsonl"
    output = tmp_path / "sft.jsonl"
    records = [
        {
            "sample_id": "group_a:task-a",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "first answer",
                    "tool_calls": [
                        {
                            "id": "first-call",
                            "type": "function",
                            "function": {"name": "Terminate", "arguments": "{}"},
                        }
                    ],
                }
            ],
            "tools": [{"type": "function", "function": {"name": "Terminate"}}],
            "sampling": {},
        },
        {
            "sample_id": "group_b:task-b",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "second answer",
                    "tool_calls": [
                        {
                            "id": "second-call",
                            "type": "function",
                            "function": {"name": "Terminate", "arguments": "{}"},
                        }
                    ],
                }
            ],
            "tools": [{"type": "function", "function": {"name": "Terminate"}}],
            "sampling": {},
        },
    ]
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    assert (
        prepare_ms_swift_dataset(
            source,
            output,
            tool_policy="preserve",
            sample_id_prefix_repeats={"group_a:": 5},
        )
        == 6
    )
    prepared = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["sample_id"] for row in prepared].count("group_a:task-a") == 5
    assert [row["sample_id"] for row in prepared].count("group_b:task-b") == 1
    assert [row["sampling"]["training_repeat_index"] for row in prepared[:5]] == [
        1,
        2,
        3,
        4,
        5,
    ]

    prepared_again = tmp_path / "sft-again.jsonl"
    assert (
        prepare_ms_swift_dataset(
            output,
            prepared_again,
            tool_policy="preserve",
            sample_id_prefix_repeats={"group_a:": 5},
        )
        == 6
    )


def test_prepare_ms_swift_dataset_folds_reasoning_for_all_domains(tmp_path):
    source = tmp_path / "accepted.jsonl"
    output = tmp_path / "sft.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "inspect"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "step-by-step",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "final_answer",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                ],
                "tools": [{"type": "function", "function": {"name": "final_answer"}}],
                "sampling": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert prepare_ms_swift_dataset(source, output, tool_policy="preserve") == 1
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["messages"][-1]["content"] == ("<think>\nstep-by-step\n</think>")
    assert "reasoning_content" not in record["messages"][-1]


def test_resumable_checkpoint_rejects_partial_save(tmp_path):
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 4}), encoding="utf-8")
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    assert resumable_ms_swift_checkpoint(tmp_path) is None

    (checkpoint / "model.safetensors").write_bytes(b"weights")
    assert resumable_ms_swift_checkpoint(tmp_path) == checkpoint


def test_mixed_provenance_metadata_has_stable_sft_columns_and_round_trips(tmp_path):
    source, output = tmp_path / "accepted.jsonl", tmp_path / "sft.jsonl"
    policy = {
        "sample_id": "policy",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "think first",
                "tool_calls": [
                    {
                        "id": "call",
                        "type": "function",
                        "function": {
                            "name": "final_answer",
                            "arguments": "{}",
                        },
                    }
                ],
            }
        ],
        "tools": [{"type": "function", "function": {"name": "final_answer"}}],
        "sampling": {"quality_tier": "strict", "attempt": 1, "model": "policy"},
    }
    other = deepcopy(policy)
    other.update(sample_id="other", source_model="other", source_api_mode="chat_completions")
    other["sampling"].update(quality_tier="verified_repaired", phase="policy", model="other")
    other["domain_specific"] = {"nested": [{"score": 1, "extra": True}]}
    rows = [policy, other]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original = source.read_bytes()
    assert prepare_ms_swift_dataset(source, output) == 2
    prepared = [json.loads(line) for line in output.read_text().splitlines()]
    assert prepared[0].keys() == prepared[1].keys()
    assert prepared[0]["sampling"].keys() == prepared[1]["sampling"].keys()
    assert json.loads(prepared[1]["metadata_json"])["source_model"] == "other"
    assert json.loads(prepared[1]["metadata_json"])["domain_specific"] == other["domain_specific"]
    assert prepared[1]["tools"] == other["tools"]
    assert prepared[1]["messages"][0]["tool_calls"] == other["messages"][0]["tool_calls"]
    assert prepared[1]["messages"][0]["content"] == "<think>\nthink first\n</think>"
    again = tmp_path / "again.jsonl"
    assert prepare_ms_swift_dataset(output, again) == 2
    assert [json.loads(line) for line in again.read_text().splitlines()] == prepared
    assert source.read_bytes() == original

    datasets = pytest.importorskip("datasets")
    # Force separate Arrow chunks so the late-only metadata appears late,
    # reproducing the failure that a one-chunk Dataset.from_list would miss.
    loaded = datasets.load_dataset(
        "json",
        data_files=str(output),
        split="train",
        chunksize=256,
        cache_dir=str(tmp_path / "datasets_cache"),
    )
    assert len(loaded) == 2
    assert json.loads(loaded[1]["metadata_json"])["source_model"] == "other"


def test_failed_preparation_does_not_replace_existing_dataset(tmp_path):
    source, target = tmp_path / "accepted.jsonl", tmp_path / "sft.jsonl"
    source.write_text('{"sample_id":"bad"}\n')
    target.write_text("previous prepared artifact\n")
    with pytest.raises(ValueError, match="missing messages"):
        prepare_ms_swift_dataset(source, target)
    assert target.read_text() == "previous prepared artifact\n"
    before = source.read_bytes()
    with pytest.raises(ValueError, match="missing messages"):
        prepare_ms_swift_dataset(source, source)
    assert source.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))
