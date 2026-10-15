import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from src.domains.who_when.rejection_sampling import (
    accept_rollout,
    balance_training_rollouts,
    collect_accepted,
    curated_record,
)
from src.domains.who_when.runtime.rollout import Rollout, run_rollout
from src.reject_sampling import policy_attempts
from src.traces import AgentTrace, DetectionResult
from src.training.sampling import accepted_record, write_accepted_jsonl


class ScriptedModel:
    def __init__(self, decisions):
        self.decisions = iter(decisions)

    def preflight(self):
        return 200

    def complete(self, messages, tools):
        tool_messages = [message for message in messages if message["role"] == "tool"]
        if not tool_messages or tool_messages[-1].get("name") == "final_answer":
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "inspect-call",
                        "type": "function",
                        "function": {
                            "name": "inspect",
                            "arguments": json.dumps({"step_id_list": [1]}),
                        },
                    }
                ],
            }
        decision = next(self.decisions)
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "final-call",
                    "type": "function",
                    "function": {
                        "name": "final_answer",
                        "arguments": json.dumps(
                            {
                                "is_failure": True,
                                "evidence_step_ids": [1 if decision else 2],
                                "responsible_agent": "agent" if decision else "other",
                            }
                        ),
                    },
                }
            ],
        }


def make_trace():
    return AgentTrace.from_record(
        {
            "sample_id": "sample-1",
            "task": "task",
            "trajectory": [{"step": 1, "content": "bad result"}],
            "is_failure": True,
            "responsible_agent": "agent",
            "decisive_step": 1,
        }
    )


def _write_balancing_trace(path, sample_id, *, length, decisive_step, agent_count):
    agents = [f"agent-{index}" for index in range(agent_count)]
    record = {
        "sample_id": sample_id,
        "task": "task",
        "trajectory": [
            {
                "step": step,
                "agent_name": agents[(step - 1) % agent_count],
                "content": f"step {step}",
            }
            for step in range(1, length + 1)
        ],
        "is_failure": True,
        "responsible_agent": agents[(decisive_step - 1) % agent_count],
        "decisive_step": decisive_step,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def _accepted_for_balance(sample_id, *, attempt=1):
    call = {
        "id": f"final-{sample_id}",
        "type": "function",
        "function": {"name": "final_answer", "arguments": {"is_failure": True}},
    }
    return accepted_record(
        sample_id=sample_id,
        messages=[
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "reasoning",
                "tool_calls": [call],
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "parameters": {"type": "object"},
                },
            }
        ],
        sampling={
            "attempt": attempt,
            "quality_tier": "strict",
        },
    )


def test_whowhen_balance_repeats_only_hard_and_improves_val_fit(tmp_path):
    train = tmp_path / "train.jsonl"
    reference = tmp_path / "val.jsonl"
    accepted = tmp_path / "accepted.jsonl"
    output = tmp_path / "balanced.jsonl"
    for index in range(8):
        _write_balancing_trace(
            train,
            f"easy-{index}",
            length=4,
            decisive_step=4,
            agent_count=1,
        )
    _write_balancing_trace(train, "hard", length=12, decisive_step=1, agent_count=4)
    _write_balancing_trace(train, "hard-retry", length=12, decisive_step=1, agent_count=4)
    for index in range(4):
        _write_balancing_trace(
            reference,
            f"reference-{index}",
            length=12,
            decisive_step=1,
            agent_count=4,
        )
    write_accepted_jsonl(
        [
            *[_accepted_for_balance(f"easy-{index}") for index in range(8)],
            _accepted_for_balance("hard", attempt=2),
            _accepted_for_balance("hard-retry", attempt=3),
        ],
        accepted,
    )

    balance_training_rollouts(
        accepted_path=accepted,
        data_path=train,
        reference_path=reference,
        output_path=output,
        seed=7,
        max_growth_fraction=0.5,
        max_repeats_per_record=3,
    )

    selected = [json.loads(line) for line in output.read_text().splitlines()]
    selected_counts = Counter(row["sample_id"] for row in selected)
    stats = json.loads(output.with_suffix(".stats.json").read_text())
    assert all(selected_counts[f"easy-{index}"] == 1 for index in range(8))
    assert selected_counts["hard"] > 1
    assert selected_counts["hard-retry"] > 1
    assert len(selected) == 15
    assert stats["original_records_preserved"] == 10
    assert stats["repeated_hard_records"] == 5
    assert stats["eligible_hard"] == 2
    assert stats["weighted_tv_after"] < stats["weighted_tv_before"]


def test_shared_policy_attempts_stop_after_enough_accepted():
    records, diagnostics = policy_attempts(
        attempts_per_item=3,
        accepted_per_item=1,
        policy_attempt=lambda attempt: ({"attempt": attempt}, {}) if attempt == 2 else (None, {}),
    )

    assert records == [{"attempt": 2}]
    assert [row["phase"] for row in diagnostics] == ["policy", "policy"]
    assert [row["attempt"] for row in diagnostics] == [1, 2]


def test_domain_directory_sampler_preserves_strict_acceptance_and_resume(monkeypatch, tmp_path):
    from src.domains.base import SamplingRequest
    from src.domains.who_when import rejection_sampling as module

    model = ScriptedModel([True])
    monkeypatch.setattr(module, "OpenAICompatibleClient", lambda **kwargs: model)
    monkeypatch.setattr(module, "iter_traces", lambda *args: iter([make_trace()]))
    request = SamplingRequest(
        Path(__file__).resolve().parents[1] / "configs/who_when.yaml",
        tmp_path / "input.jsonl",
        tmp_path,
        "policy",
        "unused",
        "initial_harness",
        1,
        1,
        1,
        10,
        "strict",
    )
    stats = module.collect(request)
    assert stats.accepted == 1
    assert stats.strict_accepted == 1
    assert stats.repaired_accepted == 0
    assert stats.attempts == 1
    records = [
        json.loads(line) for line in (tmp_path / "accepted_rollouts.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["sampling"]["inspected_step_ids"] == [1]
    resumed = module.collect(replace(request, resume=True))
    assert resumed.attempts == resumed.accepted == 0
    assert resumed.skipped_existing == 1


def test_rollout_executes_inspect_then_final_answer():
    trace = make_trace()
    rollout = run_rollout(trace, ScriptedModel([True]))
    assert accept_rollout(rollout, trace)
    assert rollout.inspected_step_ids == [1]
    assert rollout.turns == 2


def test_rollout_executes_only_first_parallel_tool_call():
    class ParallelModel:
        def __init__(self):
            self.turn = 0

        def complete(self, messages, tools):
            self.turn += 1
            if self.turn == 1:
                return {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "inspect-1",
                            "type": "function",
                            "function": {
                                "name": "inspect",
                                "arguments": json.dumps({"step_id_list": [1]}),
                            },
                        },
                        {
                            "id": "inspect-2",
                            "type": "function",
                            "function": {
                                "name": "inspect",
                                "arguments": json.dumps({"step_id_list": [2]}),
                            },
                        },
                    ],
                }
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "final",
                        "type": "function",
                        "function": {
                            "name": "final_answer",
                            "arguments": json.dumps(
                                {
                                    "is_failure": True,
                                    "evidence_step_ids": [1],
                                    "responsible_agent": "agent",
                                }
                            ),
                        },
                    }
                ],
            }

    rollout = run_rollout(make_trace(), ParallelModel())

    assert rollout.inspected_step_ids == [1]
    tool_messages = [m for m in rollout.messages if m["role"] == "tool"]
    assert json.loads(tool_messages[1]["content"])["retryable"] is True
    assert rollout.result is not None
    assert len(rollout.interface_trace["requests"]) == 2
    assert len(rollout.interface_trace["requests"][0]["response"]["tool_calls"]) == 2
    assert rollout.interface_trace["termination"] == "submitted"


def test_rejection_sampling_drops_wrong_attempt():
    attempts = []
    records, stats = collect_accepted(
        [make_trace()],
        ScriptedModel([False, True]),
        attempts_per_item=2,
        quality_tier="strict",
        on_attempt=attempts.append,
    )
    assert len(records) == 1
    assert records[0]["sampling"]["attempt"] == 2
    assert [tool["function"]["name"] for tool in records[0]["tools"]] == [
        "inspect",
        "final_answer",
    ]
    assert stats.attempts == 2
    assert stats.accepted == 1
    assert [attempt["score"] for attempt in attempts] == [0.0, 1.0]
    assert [attempt["items"] for attempt in attempts] == [1, 1]
    assert [attempt["attempts_total"] for attempt in attempts] == [2, 2]
    assert [attempt["accepted_total"] for attempt in attempts] == [1, 1]


def test_balanced_sampling_does_not_repair_wrong_final_label():
    records, stats = collect_accepted(
        [make_trace()],
        ScriptedModel([False]),
        attempts_per_item=1,
        quality_tier="balanced",
    )
    assert records == []
    assert stats.accepted == stats.repaired_accepted == 0


def test_strict_acceptance_requires_exact_step_and_agent():
    trace = make_trace()
    rollout = Rollout(
        trace.sample_id,
        [],
        DetectionResult(True, (10,), responsible_agent="agent-helper"),
        [1],
        1,
    )

    assert not accept_rollout(rollout, trace)


def test_missing_final_answer_is_rejected_without_raising():
    trace = make_trace()
    rollout = Rollout(
        trace.sample_id,
        [{"role": "assistant", "content": "answer without tool call"}],
        DetectionResult(True, (1,), responsible_agent="agent"),
        [1],
        1,
    )
    record, reason = curated_record(
        rollout,
        trace,
        attempt=1,
        quality_tier="balanced",
        require_inspection=True,
    )
    assert record is None
    assert reason == "missing_final_answer_tool_call"


def test_partial_step_label_is_used_without_inventing_an_agent():
    trace = AgentTrace.from_record(
        {
            "sample_id": "step-only",
            "task": "task",
            "trajectory": [{"step": 1, "content": "bad"}],
            "is_failure": True,
            "decisive_step": 1,
        }
    )
    rollout = Rollout(
        trace.sample_id,
        [],
        DetectionResult(True, (1,), responsible_agent=""),
        [1],
        1,
    )
    assert accept_rollout(rollout, trace)


def test_safe_detection_rejects_unsupported_attribution():
    trace = AgentTrace.from_record(
        {
            "sample_id": "safe",
            "task": "task",
            "trajectory": [{"step": 1, "content": "ok"}],
            "is_failure": False,
        }
    )
    clean = Rollout(trace.sample_id, [], DetectionResult(False), [1], 1)
    invented = Rollout(
        trace.sample_id,
        [],
        DetectionResult(False, (1,), responsible_agent="agent"),
        [1],
        1,
    )
    assert accept_rollout(clean, trace)
    assert not accept_rollout(invented, trace)
