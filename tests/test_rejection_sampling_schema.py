from __future__ import annotations

import json
import re

import pytest

from src.domains.who_when.rejection_sampling import curated_record
from src.domains.who_when.runtime.rollout import Rollout
from src.traces import AgentTrace, DetectionResult
from src.training.message_schema import validate_training_record
from src.training.ms_swift import prepare_ms_swift_dataset
from src.training.sampling import accepted_record, read_accepted_jsonl
from harnesses.rbac.initial_harness import build_harness as build_rbac_harness


def _call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _who_when_record() -> dict:
    trace = AgentTrace.from_record(
        {
            "sample_id": "who:1",
            "task": "find the failure",
            "trajectory": [{"step": 1, "content": "bad"}],
            "is_failure": True,
            "decisive_step": 1,
            "responsible_agent": "worker",
        }
    )
    messages = [
        {"role": "system", "content": "inspect evidence"},
        {"role": "user", "content": "trace"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "inspect the suspicious step",
            "tool_calls": [_call("who-inspect", "inspect", {"step_id_list": [1]})],
        },
        {
            "role": "tool",
            "tool_call_id": "who-inspect",
            "name": "inspect",
            "content": "bad",
        },
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "the inspected step proves the failure",
            "tool_calls": [
                _call(
                    "who-final",
                    "final_answer",
                    {
                        "is_failure": True,
                        "evidence_step_ids": [1],
                        "responsible_agent": "worker",
                        "rationale": "bad output",
                    },
                )
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "who-final",
            "name": "final_answer",
            "content": "accepted",
        },
    ]
    rollout = Rollout(
        trace.sample_id,
        messages,
        DetectionResult(True, (1,), responsible_agent="worker"),
        [1],
        2,
    )
    record, reason = curated_record(
        rollout, trace, attempt=1, quality_tier="strict", require_inspection=True
    )
    assert reason == "strict_correct" and record is not None
    return record


def _rbac_record() -> dict:
    harness = build_rbac_harness([])

    class Model:
        @staticmethod
        def complete(messages, tools):
            return {
                "role": "assistant",
                "content": "",
                "reasoning_content": "deny",
                "tool_calls": [_call("rbac-final", "final_answer", {"access_granted": False})],
            }

    route = harness.route("Role: unknown\nAccessible Columns: t: id", "unknown")
    result = harness.call_llm(Model(), "select secret", "unknown", route)
    return accepted_record(
        sample_id="rbac:1",
        messages=result.messages,
        tools=harness.tools,
        sampling={"quality_tier": "strict", "curation_reason": "binary_match"},
    )


@pytest.mark.parametrize(
    "record_factory",
    [_who_when_record, _rbac_record],
    ids=["who_when", "rbac"],
)
def test_all_domains_share_canonical_rejection_sampling_schema(record_factory, tmp_path):
    record = record_factory()
    validate_training_record(record)
    declared = {tool["function"]["name"] for tool in record["tools"]}
    assistants = [m for m in record["messages"] if m["role"] == "assistant"]
    assert assistants
    for message in assistants:
        assert isinstance(message["content"], str)
        assert isinstance(message["reasoning_content"], str)
        assert len(message["tool_calls"]) == 1
        assert message["tool_calls"][0]["function"]["name"] in declared
        assert not re.search(r"(?im)^\s*(Thought|Action)\s*:", message["content"])

    source = tmp_path / "accepted.jsonl"
    output = tmp_path / "sft.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert prepare_ms_swift_dataset(source, output, tool_policy="preserve") == 1
    prepared = json.loads(output.read_text(encoding="utf-8"))
    for message in prepared["messages"]:
        if message["role"] == "assistant":
            assert "reasoning_content" not in message
            assert message["content"].startswith("<think>")


def test_schema_rejects_legacy_text_actions_and_undeclared_tools():
    record = _rbac_record()
    assistant = next(m for m in record["messages"] if m["role"] == "assistant")
    assistant["content"] = "Thought: done\nAction: final_answer(...)"
    with pytest.raises(ValueError, match="legacy Thought/Action"):
        validate_training_record(record)

    record = _rbac_record()
    record["tools"] = [
        tool for tool in record["tools"] if tool["function"]["name"] != "final_answer"
    ]
    with pytest.raises(ValueError, match="undeclared tools"):
        validate_training_record(record)


def test_schema_rejects_user_message_after_tool_result():
    record = _rbac_record()
    record["messages"].extend(
        [
            {"role": "user", "content": "Do not repeat the same tool call."},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "finish",
                "tool_calls": [_call("rbac-final-2", "final_answer", {"access_granted": False})],
            },
        ]
    )

    with pytest.raises(ValueError, match="user role immediately after a tool result"):
        validate_training_record(record)


def test_resume_rejects_legacy_accepted_data_before_counting_it(tmp_path):
    path = tmp_path / "accepted_rollouts.jsonl"
    record = _rbac_record()
    record["messages"][-1] = {
        "role": "assistant",
        "content": "Thought: done\nAction: Terminate(output='answer.csv')",
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid accepted rollout"):
        read_accepted_jsonl(path)
