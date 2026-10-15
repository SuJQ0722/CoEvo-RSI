import json

import pytest
from jsonschema import ValidationError

from src.training import curation_agent as agent
from src.utils.persistence import atomic_jsonl


@pytest.fixture
def staged(tmp_path):
    files = {
        name: tmp_path / name for name in ["catalog.jsonl", "records.jsonl", "val_questions.jsonl"]
    }
    atomic_jsonl(
        files["catalog.jsonl"],
        [
            {
                "record_id": "r0",
                "hard": True,
                "features": {"source": "A"},
                "assistant_characters": 300,
                "question": "hard question",
            },
            {
                "record_id": "r1",
                "hard": False,
                "features": {"source": "B"},
                "assistant_characters": 50,
                "question": "easy question",
            },
        ],
    )
    atomic_jsonl(files["records.jsonl"], [{"record_id": "r0", "record": {"messages": []}}])
    atomic_jsonl(files["val_questions.jsonl"], [{"question": "validation question without label"}])
    return files, {"min_records": 5, "max_records": 10, "previous_errors": []}


def test_data_tools_bound_reads_and_protect_hard_rows(staged):
    files, inputs = staged
    session = agent.CurationTools(files, inputs)
    with pytest.raises(ValidationError):
        session.call("read_data", {"file": "../../.env"})
    with pytest.raises(ValueError, match="unknown tool"):
        session.call("shell", {"command": "cat /etc/passwd"})
    with pytest.raises(ValueError, match="protected hard"):
        session.call("set_counts", {"filter": {}, "count": 0})
    assert session.counts == {}
    with pytest.raises(ValueError, match="only hard"):
        session.call("set_counts", {"filter": {"hard": False}, "count": 2})
    with pytest.raises(ValueError, match="unknown record"):
        session.call("set_counts", {"filter": {"record_ids": ["missing"]}, "count": 1})
    with pytest.raises(ValueError, match="unknown feature"):
        session.call("inspect_catalog", {"filter": {"features": {"secret": "x"}}})
    chunk = session.call("read_data", {"file": "val_questions.jsonl", "length": 8})
    assert len(chunk["text"]) == 8 and chunk["next_offset"] == 8 and not chunk["eof"]
    result = session.call("set_counts", {"filter": {"features": {"source": "A"}}, "count": 4})
    assert result["mixture"]["records"] == 5
    assert result["mixture"]["features"]["source"] == {"A": 4, "B": 1}
    session.call("submit_plan", {"rationale": "Preserve and repeat the hard cohort"})
    assert session.plan["counts"] == {"r0": 4}


def test_mixture_distinguishes_questions_trajectories_and_repetition(staged):
    files, inputs = staged
    session = agent.CurationTools(files, inputs)
    first, second = session.catalog
    first["sample_id"] = second["sample_id"] = "same-question"
    session.call("set_counts", {"filter": {"record_ids": ["r0"]}, "count": 4})
    summary = session.summary(session.catalog, weighted=True)
    assert (summary["records"], summary["unique_records"], summary["unique_samples"],
            summary["repeated_records"]) == (5, 2, 1, 3)
    session.call("set_counts", {"filter": {"record_ids": ["r1"]}, "count": 0})
    summary = session.summary(session.catalog, weighted=True)
    assert (summary["records"], summary["unique_records"], summary["unique_samples"],
            summary["repeated_records"]) == (4, 1, 1, 3)
    filtered = session.call("inspect_catalog", {"filter": {"features": {"source": "A"}}})
    assert filtered["summary"]["records"] == filtered["summary"]["unique_samples"] == 1


def test_agent_corrects_tool_errors_and_publishes_plan_without_process(
    staged, tmp_path, monkeypatch
):
    files, inputs = staged

    def forbidden_process(*args, **kwargs):
        raise AssertionError("data curation must not start a shell or Landlock process")

    monkeypatch.setattr("subprocess.run", forbidden_process)
    monkeypatch.setattr("subprocess.Popen", forbidden_process)
    actions = [
        ("inspect_catalog", {}),
        ("submit_plan", {"rationale": "Too small"}),
        ("set_counts", {"filter": {"hard": True}, "count": 4}),
        ("submit_plan", {"rationale": "Repeat hard examples to meet minimum"}),
    ]
    observed = []

    class Client:
        max_tokens = 16384

        def complete(self, messages, tools):
            observed.append(messages)
            name, args = actions[len(observed) - 1]
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": str(len(observed)),
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }

    monkeypatch.setattr(agent, "build_client", lambda config: Client())
    result = agent.run_curation_agent(
        config={}, output_dir=tmp_path / "attempt", inputs=inputs, input_files=files
    )
    assert "physical record count" in observed[2][-1]["content"]
    assert json.loads(result.read_text())["plan_file"] == "plan.json"
    assert json.loads((result.parent / "plan.json").read_text())["counts"] == {"r0": 4}
    assert "You may use scripts" not in observed[0][0]["content"]


def test_empty_agent_output_has_bounded_turn_budget(staged, tmp_path, monkeypatch):
    files, inputs = staged

    class Client:
        max_tokens = 16384

        def complete(self, messages, tools):
            return {"role": "assistant", "content": ""}

    monkeypatch.setattr(agent, "build_client", lambda config: Client())
    with pytest.raises(RuntimeError, match="in 2 turns"):
        agent.run_curation_agent(
            config={"training": {"data_curation": {"agent_max_turns": 2}}},
            output_dir=tmp_path / "attempt",
            inputs=inputs,
            input_files=files,
        )
    assert not (tmp_path / "attempt/workspace/plan.json").exists()


def test_data_agent_uses_same_meta_provider_credentials_and_protocol(monkeypatch):
    monkeypatch.setenv("TEST_META_KEY", "test-secret")
    client = agent.build_client(
        {
            "meta_evolver": {
                "backend": "codex",
                "codex": {
                    "model": "meta-model",
                    "base_url": "https://configured.invalid/v1",
                    "api_key_env": "TEST_META_KEY",
                    "reasoning_effort": "high",
                },
            }
        }
    )
    assert client.model == "meta-model" and client.api_mode == "responses"
    assert client.api_key == "test-secret" and client.reasoning_effort == "high"
    assert client.max_tokens == 16384


@pytest.mark.parametrize("explicit_reason", [False, True])
def test_agent_recovers_exhausted_reasoning_budget(staged, tmp_path, monkeypatch, explicit_reason):
    files, inputs = staged
    budgets = []

    class Client:
        max_tokens = 8

        def complete(self, messages, tools):
            budgets.append(self.max_tokens)
            if len(budgets) == 1:
                if explicit_reason:
                    return {"_coevox_incomplete_reason": "max_output_tokens"}
                return {"_coevox_usage": {"completion_tokens": 8}}
            return {
                "tool_calls": [
                    {"id": "set", "function": {"name": "set_counts", "arguments":
                        json.dumps({"filter": {"hard": True}, "count": 4})}},
                    {"id": "submit", "function": {"name": "submit_plan", "arguments":
                        json.dumps({"rationale": "Retain all hard rows and meet minimum"})}},
                ]
            }

    monkeypatch.setattr(agent, "build_client", lambda config: Client())
    result = agent.run_curation_agent(
        config={"training": {"data_curation": {"agent_max_tokens_cap": 16}}},
        output_dir=tmp_path / "attempt", inputs=inputs, input_files=files,
    )
    assert result.exists() and budgets == [8, 16]
    events = json.loads((tmp_path / "attempt/events.json").read_text())
    assert events[0]["recovery"] == {"max_tokens": 16}


def test_exhausted_budget_has_a_hard_cap(staged, tmp_path, monkeypatch):
    files, inputs = staged

    class Client:
        max_tokens = 16

        def complete(self, messages, tools):
            return {"_coevox_incomplete_reason": "max_output_tokens"}

    monkeypatch.setattr(agent, "build_client", lambda config: Client())
    with pytest.raises(RuntimeError, match="exhausted output token budget 16"):
        agent.run_curation_agent(
            config={"training": {"data_curation": {"agent_max_tokens_cap": 16}}},
            output_dir=tmp_path / "attempt", inputs=inputs, input_files=files,
        )
    assert not (tmp_path / "attempt/workspace/plan.json").exists()
