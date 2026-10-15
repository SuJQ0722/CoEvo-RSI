import json

import pytest

from harnesses.who_when.loader import build_harness
from harnesses.who_when.initial_harness import FailureDetectionHarness
from harnesses.who_when.initial_harness.tools.inspect import (
    MAX_TOOL_RESPONSE_CHARS,
    TRUNCATION_NOTICE,
)
from src.traces import AgentTrace


def make_trace(is_failure=True):
    return AgentTrace.from_record(
        {
            "sample_id": "sample-1",
            "source_dataset": "unit",
            "task": "Do the task",
            "is_failure": is_failure,
            "decisive_step": 2,
            "trajectory": [
                {"step": 1, "role": "assistant", "content": "look around"},
                {"step": 2, "role": "tool", "content": "SECRET FULL PAYLOAD"},
            ],
        }
    )


def test_initial_context_is_structured_summary_not_full_trace():
    harness = FailureDetectionHarness(make_trace())
    messages = harness.initial_messages()
    payload = json.loads(messages[1]["content"])

    assert "trajectory_summary" in payload
    assert "SECRET FULL PAYLOAD" not in messages[1]["content"]
    assert "is_failure" not in messages[1]["content"]
    assert "sample-1" not in messages[1]["content"]
    assert "unit" not in messages[1]["content"]
    assert payload["trajectory_summary"]["steps"][1]["step_id"] == 2


def test_harness_exposes_exactly_two_tools_and_inspects_selected_steps():
    harness = FailureDetectionHarness(make_trace())
    assert [tool["function"]["name"] for tool in harness.tools] == [
        "inspect",
        "final_answer",
    ]

    response = json.loads(
        harness.handle_tool_call("inspect", {"step_id_list": [2, 99]})
    )
    assert response["steps"][0]["record"]["content"] == "SECRET FULL PAYLOAD"
    assert response["steps"][1]["error"] == "step not found"
    assert harness.inspected_step_ids == [2]

    harness.handle_tool_call(
        "final_answer",
        {"is_failure": True, "evidence_step_ids": [2], "rationale": "bad tool output"},
    )
    assert harness.is_done
    assert harness.result is not None and harness.result.is_failure


def test_inspect_truncates_oversized_tool_response_as_valid_json():
    trace = AgentTrace.from_record(
        {
            "sample_id": "large-step",
            "task": "task",
            "trajectory": [{"step": 1, "content": "useful-prefix-" + "x" * 30_000}],
            "is_failure": False,
        }
    )
    response = FailureDetectionHarness(trace).handle_tool_call(
        "inspect", {"step_id_list": [1]}
    )
    payload = json.loads(response)

    assert len(response) <= MAX_TOOL_RESPONSE_CHARS
    assert payload["truncated"] is True
    assert "useful-prefix-" in payload["content"]
    assert payload["content"].endswith(TRUNCATION_NOTICE)


def test_failure_attribution_rejects_empty_final_fields_and_allows_retry():
    trace = AgentTrace.from_record(
        {
            "sample_id": "attribution",
            "task": "task",
            "trajectory": [{"step": 1, "agent_name": "agent", "content": "bad"}],
            "is_failure": True,
            "decisive_step": 1,
            "responsible_agent": "agent",
        }
    )
    harness = FailureDetectionHarness(trace)
    final_schema = harness.tools[1]["function"]["parameters"]["properties"]
    assert final_schema["evidence_step_ids"]["minItems"] == 1
    assert final_schema["responsible_agent"]["minLength"] == 1

    response = json.loads(
        harness.handle_tool_call(
            "final_answer",
            {"is_failure": True, "evidence_step_ids": [], "responsible_agent": ""},
        )
    )
    assert response["retryable"] is True
    assert response["missing"] == ["evidence_step_ids", "responsible_agent"]
    assert not harness.is_done

    harness.handle_tool_call(
        "final_answer",
        {
            "is_failure": True,
            "evidence_step_ids": [1],
            "responsible_agent": "agent",
        },
    )
    assert harness.is_done


def test_failure_detection_keeps_attribution_fields_optional():
    trace = AgentTrace.from_record(
        {
            "sample_id": "detection",
            "task": "task",
            "trajectory": [],
            "is_failure": False,
        }
    )
    harness = FailureDetectionHarness(trace)
    final_schema = harness.tools[1]["function"]["parameters"]["properties"]
    assert "minItems" not in final_schema["evidence_step_ids"]
    assert "minLength" not in final_schema["responsible_agent"]

    harness.handle_tool_call(
        "final_answer",
        {"is_failure": False, "evidence_step_ids": [], "responsible_agent": ""},
    )
    assert harness.is_done


def test_harness_loader_preserves_initial_harness_and_rejects_invalid_names():
    harness = build_harness("initial_harness", make_trace())
    assert isinstance(harness, FailureDetectionHarness)
    with pytest.raises(ValueError):
        build_harness("../outside", make_trace())
