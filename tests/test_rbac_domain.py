from __future__ import annotations

import json
from pathlib import Path

from src.config import load_config
from src.domains import EvaluationResult, get_domain_plugin
from src.domains.rbac.data import RBACSample, load, load_policy_rows
from src.domains.rbac.evaluator import evaluate
from src.domains.rbac.rejection_sampling import (
    build_balanced_training_rollouts,
    collect_to_directory,
)
from src.domains.rbac.runtime.infer import evaluate_one
from src.domains.rbac.smoke import test as smoke_test
from src.training.ms_swift import prepare_ms_swift_dataset
from src.training.preparation import prepare_training_dataset
from harnesses.rbac.generated import (
    build_harness as build_repaired_rbac_harness,
)
from harnesses.rbac.generated.tools.final_answer import (
    execute as execute_repaired_final_answer,
)
from harnesses.rbac.initial_harness.tools.final_answer import (
    SCHEMA as RBAC_FINAL_ANSWER,
)
from harnesses.rbac.initial_harness.tools.final_answer import (
    execute as execute_seed_final_answer,
)
from harnesses.rbac.loader import build_harness

ROOT = Path(__file__).resolve().parents[1]


def test_rbac_smoke_allows_candidate_entrypoint_evolution():
    report = smoke_test(
        "generated",
        package="harnesses.rbac",
        root=ROOT / "harnesses" / "rbac",
    )
    assert report.passed
    assert "stable_entrypoint" not in {check.name for check in report.checks}


def test_final_answer_accepts_canonical_string_booleans_only():
    for execute in (execute_seed_final_answer, execute_repaired_final_answer):
        _, marker, sql = execute({"access_granted": " True ", "sql": " SELECT 1 "})
        assert (marker, sql) == ("ACCESS_GRANTED", "SELECT 1")
        _, marker, sql = execute({"access_granted": "false"})
        assert (marker, sql) == ("ACCESS_DENIED", None)

        response, marker, sql = execute({"access_granted": "yes"})
        assert (marker, sql) == (None, None)
        assert json.loads(response)["error"] == "access_granted must be a boolean"


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_seed_harness_is_text2sql_conservative_and_binary():
    policy = [
        {"instruction": "CEO secret", "role": "CEO (t)", "test_role": "CEO (t)", "valid": 1},
        {
            "instruction": "Team note",
            "role": "CEO (t) - IT (s)",
            "test_role": "CEO (t) - IT (s)",
            "valid": 1,
        },
    ]
    harness = build_harness("initial_harness", policy)
    full = "Total Tables: 1\nRole: Analyst\nAccessible Columns: t: ALL COLUMNS"
    partial = "Total Tables: 1\nRole: NewRole\nAccessible Columns: t: id"
    assert harness.route(full, "Analyst").decision == 1
    assert harness.route(full, "OtherRole").decision == 0
    assert harness.route(partial, "NewRole").decision is None
    assert harness.route("", "").decision == 0
    assert harness.parse_access("ACCESS_DENIED") == 0
    assert harness.parse_access("ACCESS_GRANTED") == 1
    assert harness.parse_access("Here is the answer") == 0
    assert [tool["function"]["name"] for tool in harness.tools] == ["final_answer", "test_sql"]

    class FakeModel:
        @staticmethod
        def complete(messages, tools):
            assert tools[0] == RBAC_FINAL_ANSWER
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "final_answer",
                            "arguments": json.dumps({"access_granted": True}),
                        },
                    }
                ],
                "_coevox_usage": {"prompt_tokens": 5, "total_tokens": 8},
            }

    result = harness.call_llm(
        FakeModel(),
        partial,
        "NewRole",
        harness.route(partial, "NewRole"),
    )
    assert result.response == "ACCESS_GRANTED"
    assert result.usage == {"prompt_tokens": 5, "total_tokens": 8}
    assert result.messages[-2]["tool_calls"][0]["function"]["name"] == "final_answer"
    assert result.messages[-1]["role"] == "tool"


def test_four_metrics_and_binary_per_sample_score():
    records = [
        {
            "sample_id": "1",
            "prediction": {"access": 1, "rule_decision": 1},
            "gold": {"valid": 1},
            "llm_called": 1,
            "rule_used": 1,
            "test_type": "pos",
            "db_id": "db_a",
            "difficulty": "easy",
        },
        {
            "sample_id": "2",
            "prediction": {"access": 0, "rule_decision": 0},
            "gold": {"valid": 0},
            "llm_called": 0,
            "rule_used": 1,
            "test_type": "neg",
            "db_id": "db_a",
            "difficulty": "easy",
        },
        {
            "sample_id": "3",
            "prediction": {"access": 0, "rule_decision": None},
            "gold": {"valid": 1},
            "llm_called": 1,
            "rule_used": 0,
            "test_type": "pos",
            "db_id": "db_b",
            "difficulty": "hard",
        },
    ]
    result = evaluate(records)
    assert result.metrics["ac_f1"] == 2 / 3
    assert result.metrics["safe_ex"] == 0.0
    assert result.metrics["llm_rate"] == 2 / 3
    assert result.metrics["rule_rate"] == 2 / 3
    assert result.metrics["rule_accuracy"] == 1.0
    assert [item["score"] for item in result.per_sample] == [0, 1, 0]
    assert all(item["score"] in {0, 1} for item in result.per_sample)
    assert set(result.diagnostics["per_database"]) == {"db_a", "db_b"}
    assert set(result.diagnostics["per_difficulty"]) == {"easy", "hard"}
    assert result.diagnostics["protocol_invalid"] == 0

    malformed = evaluate(
        [
            {
                "sample_id": "bad-tool",
                "prediction": {"access": None, "rule_decision": None},
                "gold": {"valid": 0},
                "llm_called": 1,
                "rule_used": 0,
                "test_type": "neg",
                "error": "assistant returned no final_answer tool call",
            }
        ]
    )
    assert malformed.per_sample[0]["score"] == 0
    assert malformed.per_sample[0]["protocol_valid"] is False
    assert malformed.diagnostics["confusion"] == {"tp": 0, "fp": 1, "fn": 0, "tn": 0}
    assert malformed.diagnostics["six_category_counts"]["violation_wrong"] == 1


def test_rule_is_terminal_and_fallback_requires_formal_tool_call():
    policy = [
        {"instruction": "CEO secret", "role": "CEO (t)", "test_role": "CEO (t)", "valid": 1},
        {
            "instruction": "Team note",
            "role": "CEO (t) - IT (s)",
            "test_role": "CEO (t) - IT (s)",
            "valid": 1,
        },
    ]
    harness = build_harness("initial_harness", policy)

    class NoCallModel:
        def complete(self, messages, tools):
            raise AssertionError("a terminal rule must not call the model")

    allowed_instruction = "Total Tables: 1\nRole: Analyst\nAccessible Columns: t: ALL COLUMNS"
    denied_instruction = "Role: Auditor\nAccessible Columns: t: id"
    fallback_instruction = "Role: NewRole\nAccessible Columns: t: id"
    allowed = evaluate_one(
        RBACSample("allow", allowed_instruction, "Analyst", 1), harness, NoCallModel()
    )
    denied = evaluate_one(
        RBACSample("deny", denied_instruction, "Analyst", 0),
        harness,
        NoCallModel(),
    )
    assert (allowed["prediction"]["response"], allowed["llm_called"]) == (
        "ACCESS_GRANTED",
        0,
    )
    assert (denied["prediction"]["response"], denied["llm_called"]) == (
        "ACCESS_DENIED",
        0,
    )

    class ToolModel:
        @staticmethod
        def complete(messages, tools):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "final_answer",
                            "arguments": json.dumps({"access_granted": True}),
                        },
                    }
                ],
            }

    fallback = evaluate_one(
        RBACSample("fallback", fallback_instruction, "NewRole", 1), harness, ToolModel()
    )
    assert fallback["llm_called"] == 1
    assert fallback["prediction"]["response"] == "ACCESS_GRANTED"
    assert fallback["messages"][-2]["tool_calls"][0]["function"]["name"] == "final_answer"
    assert allowed["interface_trace"]["requests"] == []
    assert allowed["interface_trace"]["termination"] == "rule_only"
    assert len(fallback["interface_trace"]["requests"]) == 1
    assert fallback["interface_trace"]["requests"][0]["tools"] == harness.tools


def test_active_evolved_harness_keeps_protocol_messages_schema_valid():
    instruction = "Role: Analyst\nAccessible Columns: t: id"
    harness = build_repaired_rbac_harness([])
    route = harness.route(instruction, "Analyst")

    class MissingToolThenFinalModel:
        def __init__(self):
            self.turn = 0

        def complete(self, messages, tools):
            self.turn += 1
            if self.turn == 1:
                return {"role": "assistant", "content": "no tool"}
            return {
                "role": "assistant",
                "content": "",
                "reasoning_content": "finish",
                "tool_calls": [
                    _tool_call("missing-final", "final_answer", {"access_granted": False})
                ],
            }

    missing = harness.call_llm(MissingToolThenFinalModel(), instruction, "Analyst", route)
    missing_roles = [message["role"] for message in missing.messages]
    assert missing_roles[:4] == ["system", "user", "assistant", "user"]
    assert "Protocol correction" in missing.messages[3]["content"]
    assert all(
        not (left == "tool" and right == "user")
        for left, right in zip(missing_roles, missing_roles[1:])
    )
    assert missing_roles[-2:] == ["assistant", "tool"]

    class DuplicateSqlThenFinalModel:
        def __init__(self):
            self.turn = 0

        def complete(self, messages, tools):
            self.turn += 1
            if self.turn < 3:
                return {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "verify",
                    "tool_calls": [_tool_call(f"sql-{self.turn}", "test_sql", {"sql": "SELECT 1"})],
                }
            return {
                "role": "assistant",
                "content": "",
                "reasoning_content": "finish",
                "tool_calls": [
                    _tool_call(
                        "duplicate-final",
                        "final_answer",
                        {"access_granted": True, "sql": "SELECT 1"},
                    )
                ],
            }

    duplicate = harness.call_llm(
        DuplicateSqlThenFinalModel(),
        instruction,
        "Analyst",
        route,
        db_path=":memory:",
    )
    roles = [message["role"] for message in duplicate.messages]
    assert all(
        not (left == "tool" and right == "user")
        for left, right in zip(roles, roles[1:])
    )
    assert roles[-2:] == ["assistant", "tool"]


def test_rbac_selection_requires_f1_and_rule_accuracy_then_saves_calls():
    config = load_config(ROOT / "configs/rbac.yaml")
    plugin = get_domain_plugin(config)
    parent = EvaluationResult(
        {"safe_ex": 0.9, "ac_f1": 0.9, "llm_rate": 0.8, "rule_rate": 0.2, "rule_accuracy": 1.0}
    )
    accepted = plugin.select(
        parent,
        EvaluationResult(
            {"safe_ex": 0.9, "ac_f1": 0.9, "llm_rate": 0.7, "rule_rate": 0.3, "rule_accuracy": 1.0}
        ),
        config["selection"],
    )
    unsafe = plugin.select(
        parent,
        EvaluationResult(
            {
                "safe_ex": 0.89,
                "ac_f1": 0.9,
                "llm_rate": 0.5,
                "rule_rate": 0.5,
                "rule_accuracy": 0.99,
            }
        ),
        config["selection"],
    )
    misfilter = plugin.select(
        parent,
        EvaluationResult(
            {"safe_ex": 0.9, "ac_f1": 0.9, "llm_rate": 0.5, "rule_rate": 0.5, "rule_accuracy": 0.8}
        ),
        config["selection"],
    )
    assert accepted.accepted
    assert not unsafe.accepted
    assert not misfilter.accepted


def test_role_aware_csv_jsonl_and_plain_sft(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "instruction,output,role,test_role,valid\nq,a,CEO (t),CEO (t),1\n",
        encoding="utf-8",
    )
    sample = next(load(csv_path))
    assert sample.sample_id == "train:000001"
    assert load_policy_rows(csv_path)[0]["role"] == "CEO (t)"

    jsonl_path = tmp_path / "train.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "sample_id": "role-aware:train:000001",
                "instruction": "q",
                "role": "CEO (t)",
                "test_role": "CEO (t)",
                "valid": 1,
                "test_type": "train",
                "target": {
                    "tool": "final_answer",
                    "arguments": {"access_granted": True},
                    "normalized": "ACCESS_GRANTED",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    jsonl_sample = next(load(jsonl_path))
    assert jsonl_sample.sample_id == "role-aware:train:000001"
    assert jsonl_sample.valid == 1
    assert jsonl_sample.output == "ACCESS_GRANTED"
    assert load_policy_rows(jsonl_path)[0]["role"] == "CEO (t)"

    source = tmp_path / "accepted.jsonl"
    output = tmp_path / "prepared.jsonl"
    source.write_text(
        json.dumps(
            {
                "sample_id": "x",
                "messages": [
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "check access",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "final_answer",
                                    "arguments": json.dumps({"access_granted": True}),
                                },
                            }
                        ],
                    },
                ],
                "tools": [RBAC_FINAL_ANSWER],
                "sampling": {"score": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert prepare_ms_swift_dataset(source, output, tool_policy="preserve") == 1
    prepared = json.loads(output.read_text(encoding="utf-8"))
    assert prepared["tools"] == [RBAC_FINAL_ANSWER]


def test_rejection_sampling_scores_are_strict_binary(tmp_path, monkeypatch):
    data = tmp_path / "train.csv"
    data.write_text(
        "instruction,output,role,test_role,valid\n"
        "allow request,answer,CEO (t),CEO (t),1\n"
        "deny request,refusal,CEO (t),CEO (t),0\n",
        encoding="utf-8",
    )

    class FakeClient:
        def __init__(self, **_):
            pass

        def complete(self, messages, tools):
            content = messages[-1]["content"]
            allowed = "deny request" not in content
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "final_answer",
                            "arguments": json.dumps({"access_granted": allowed}),
                        },
                    }
                ],
            }

    monkeypatch.setattr(
        "src.domains.rbac.rejection_sampling.OpenAICompatibleClient",
        FakeClient,
    )
    output = tmp_path / "sampling"
    stats = collect_to_directory(
        config_path=ROOT / "configs/rbac.yaml",
        data_path=data,
        output_dir=output,
        policy_model="fake",
        base_url="http://unused",
        harness_name="initial_harness",
        attempts_per_item=2,
        accepted_per_item=1,
        concurrency=1,
        require_sql_execution=False,
    )
    attempts = [
        json.loads(line) for line in (output / "sampling_attempts.jsonl").read_text().splitlines()
    ]
    accepted = [
        json.loads(line) for line in (output / "accepted_rollouts.jsonl").read_text().splitlines()
    ]
    assert stats.accepted == 2
    assert {record["score"] for record in attempts} == {1}
    assert all(type(record["score"]) is int for record in attempts)
    assert all(record["sampling"]["score"] == 1 for record in accepted)
    assert all(
        [tool["function"]["name"] for tool in record["tools"]] == ["final_answer", "test_sql"]
        for record in accepted
    )
    assert all(
        record["messages"][-2]["tool_calls"][0]["function"]["name"] == "final_answer"
        for record in accepted
    )


def test_rbac_training_mix_keeps_all_accepted_and_supplements_gold(tmp_path):
    source = tmp_path / "train.jsonl"
    source_rows = []
    for label in (0, 1):
        for index in range(3):
            source_rows.append(
                {
                    "sample_id": f"source-{label}-{index}",
                    "instruction": "Database: db\nTotal Tables: 1\nTable t has columns such as c.\n"
                    "Role: Analyst\nAccessible Columns: t: c\n\n##Access Request:\nselect c",
                    "test_role": "Analyst",
                    "valid": label,
                    "gold_sql": "SELECT c FROM t",
                }
            )
    source.write_text("".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8")

    def accepted(sample_id: str, label: bool) -> dict:
        arguments = {"access_granted": label}
        if label:
            arguments["sql"] = "SELECT c FROM t"
        call = _tool_call(f"call-{sample_id}", "final_answer", arguments)
        return {
            "sample_id": sample_id,
            "messages": [
                {"role": "user", "content": "request"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "",
                    "tool_calls": [call],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call-{sample_id}",
                    "name": "final_answer",
                    "content": "submitted",
                },
            ],
            "tools": [RBAC_FINAL_ANSWER],
            "sampling": {"score": 1},
        }

    original = [accepted("kept-allow", True)] + [
        accepted(f"kept-deny-{index}", False) for index in range(3)
    ]
    accepted_path = tmp_path / "accepted.jsonl"
    accepted_path.write_text(
        "".join(json.dumps(record) + "\n" for record in original), encoding="utf-8"
    )
    output = tmp_path / "balanced.jsonl"

    report = build_balanced_training_rollouts(
        accepted_path=accepted_path,
        data_path=source,
        output_path=output,
        harness_name="initial_harness",
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert report == {
        "accepted_records": 4,
        "strict_accepted_records": 4,
        "repaired_accepted_records": 0,
        "gold_supplements": 2,
        "dropped_for_balance": 0,
        "balance_strategy": "supplement_gold",
        "allow": 3,
        "deny": 3,
        "deny_to_allow_ratio": 1.0,
        "target_deny_to_allow_ratio": 1.0,
        "source_data": str(source),
    }
    assert records[:4] == original
    assert {record["sampling"].get("quality_tier") for record in records[4:]} == {"gold"}
    assert json.loads((tmp_path / "training_mix.json").read_text()) == report

    training_dataset = tmp_path / "candidate/sft_train.jsonl"
    prepared, training_source = prepare_training_dataset(
        domain="rbac",
        quality_tier="balanced",
        accepted_rollouts=accepted_path,
        training_dataset=training_dataset,
        train_data=source,
        harness_name="initial_harness",
        sampling_config={
            "min_deny_to_allow_ratio": 1.0,
            "max_deny_to_allow_ratio": 2.0,
            "target_deny_to_allow_ratio": 1.0,
        },
        training_config={},
    )
    assert prepared == 6 > len(original)
    assert training_source == tmp_path / "candidate/balanced_rollouts.jsonl"
    assert training_source.is_file()
    assert (tmp_path / "candidate/training_mix.json").is_file()


def test_rbac_training_mix_can_downsample_without_synthetic_gold(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text("", encoding="utf-8")

    def accepted(sample_id: str, label: bool) -> dict:
        arguments = {"access_granted": label}
        if label:
            arguments["sql"] = "SELECT 1"
        call = _tool_call(f"call-{sample_id}", "final_answer", arguments)
        return {
            "sample_id": sample_id,
            "messages": [
                {"role": "user", "content": "request"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "verified reasoning",
                    "tool_calls": [call],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call-{sample_id}",
                    "name": "final_answer",
                    "content": "submitted",
                },
            ],
            "tools": [RBAC_FINAL_ANSWER],
            "sampling": {"score": 1, "quality_tier": "strict"},
        }

    records = [accepted("allow", True)] + [
        accepted(f"deny-{index}", False) for index in range(3)
    ]
    accepted_path = tmp_path / "accepted.jsonl"
    accepted_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    output = tmp_path / "balanced.jsonl"

    report = build_balanced_training_rollouts(
        accepted_path=accepted_path,
        data_path=source,
        output_path=output,
        harness_name="initial_harness",
        strategy="downsample",
    )

    balanced = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(balanced) == 2
    assert {record["sample_id"] for record in balanced} >= {"allow"}
    assert report["allow"] == report["deny"] == 1
    assert report["gold_supplements"] == 0
    assert report["dropped_for_balance"] == 2
    assert report["balance_strategy"] == "downsample"


def test_balanced_rbac_repairs_only_suffix_after_verified_gold_sql(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text(
        json.dumps(
            {
                "sample_id": "rbac:verified-prefix",
                "instruction": "Database: demo\nRole: Analyst\n\n##Access Request:\nlist names",
                "role": "Analyst",
                "test_role": "Analyst",
                "valid": 1,
                "gold_sql": "SELECT name FROM people ORDER BY name",
                "db_id": "demo",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db_dir = tmp_path / "database"
    db_path = db_dir / "demo/demo.sqlite"
    db_path.parent.mkdir(parents=True)
    import sqlite3

    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE people(name TEXT)")
        connection.execute("INSERT INTO people VALUES ('Ada')")

    class VerifiedThenWrongFinalClient:
        def __init__(self, **_):
            pass

        def complete(self, messages, tools):
            if not any(message.get("name") == "test_sql" for message in messages):
                return {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Verify the candidate SQL.",
                    "tool_calls": [
                        _tool_call(
                            "test-correct",
                            "test_sql",
                            {"sql": "SELECT name FROM people ORDER BY name"},
                        )
                    ],
                }
            return {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    _tool_call("wrong-final", "final_answer", {"access_granted": False})
                ],
            }

    monkeypatch.setattr(
        "src.domains.rbac.rejection_sampling.OpenAICompatibleClient",
        VerifiedThenWrongFinalClient,
    )
    output = tmp_path / "sampling"
    stats = collect_to_directory(
        config_path=ROOT / "configs/rbac.yaml",
        data_path=data,
        output_dir=output,
        policy_model="fake",
        base_url="http://unused",
        harness_name="initial_harness",
        attempts_per_item=1,
        accepted_per_item=1,
        concurrency=1,
        quality_tier="balanced",
        repair_strategy="verified_only",
        db_dir=str(db_dir),
    )

    record = json.loads((output / "accepted_rollouts.jsonl").read_text())
    assert (
        stats.accepted,
        stats.strict_accepted,
        stats.repaired_accepted,
    ) == (1, 0, 1)
    assert record["sampling"]["quality_tier"] == "verified_repaired"
    assert record["sampling"]["curation_reason"] == "verified_test_sql_suffix_repair"
    assert [
        call["function"]["name"]
        for message in record["messages"]
        for call in message.get("tool_calls", [])
    ] == ["test_sql", "final_answer"]


def test_balanced_rbac_repairs_only_final_answer_boolean_format(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text(
        json.dumps(
            {
                "sample_id": "rbac:format-only",
                "instruction": "Database: demo\nRole: Analyst\n\n##Access Request:\nprivate data",
                "role": "Analyst",
                "test_role": "Analyst",
                "valid": 0,
                "db_id": "demo",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class IntegerBooleanClient:
        def __init__(self, **_):
            pass

        def complete(self, messages, tools):
            return {
                "role": "assistant",
                "content": "",
                "reasoning_content": "The role lacks access.",
                "tool_calls": [
                    _tool_call("format-final", "final_answer", {"access_granted": 0})
                ],
            }

    monkeypatch.setattr(
        "src.domains.rbac.rejection_sampling.OpenAICompatibleClient",
        IntegerBooleanClient,
    )
    output = tmp_path / "sampling"
    stats = collect_to_directory(
        config_path=ROOT / "configs/rbac.yaml",
        data_path=data,
        output_dir=output,
        policy_model="fake",
        base_url="http://unused",
        harness_name="initial_harness",
        attempts_per_item=1,
        accepted_per_item=1,
        concurrency=1,
        quality_tier="balanced",
        repair_strategy="verified_only",
    )

    record = json.loads((output / "accepted_rollouts.jsonl").read_text())
    final_arguments = record["messages"][-2]["tool_calls"][0]["function"]["arguments"]
    assert stats.repaired_accepted == 1
    assert record["sampling"]["curation_reason"] == "format_only_final_repair"
    assert json.loads(final_arguments) == {"access_granted": False}


def test_quality_first_rbac_discards_unverified_semantic_error(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text(
        json.dumps(
            {
                "sample_id": "rbac:wrong-sql",
                "instruction": "Database: demo\nRole: Analyst\n\n##Access Request:\nlist names",
                "role": "Analyst",
                "test_role": "Analyst",
                "valid": 1,
                "gold_sql": "SELECT name FROM people ORDER BY name",
                "db_id": "demo",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db_dir = tmp_path / "database"
    (db_dir / "demo").mkdir(parents=True)
    (db_dir / "demo/demo.sqlite").touch()

    class WrongSqlClient:
        def __init__(self, **_):
            pass

        def complete(self, messages, tools):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "wrong-final",
                        "final_answer",
                        {"access_granted": True, "sql": "SELECT name FROM people"},
                    )
                ],
            }

    monkeypatch.setattr(
        "src.domains.rbac.rejection_sampling.OpenAICompatibleClient",
        WrongSqlClient,
    )
    monkeypatch.setattr(
        "src.domains.rbac.rejection_sampling.exec_match_safe",
        lambda *_: False,
    )
    output = tmp_path / "sampling"
    stats = collect_to_directory(
        config_path=ROOT / "configs/rbac.yaml",
        data_path=data,
        output_dir=output,
        policy_model="fake",
        base_url="http://unused",
        harness_name="initial_harness",
        attempts_per_item=2,
        accepted_per_item=1,
        concurrency=1,
        quality_tier="balanced",
        repair_strategy="verified_only",
        db_dir=str(db_dir),
    )

    assert stats.accepted == 0
    assert (output / "accepted_rollouts.jsonl").read_text() == ""


def test_quality_first_rbac_rejects_allow_without_sql_execution(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text(
        json.dumps(
            {
                "sample_id": "rbac:no-database",
                "instruction": "Database: absent\nRole: Analyst\n\n##Access Request:\nlist names",
                "role": "Analyst",
                "test_role": "Analyst",
                "valid": 1,
                "gold_sql": "SELECT name FROM people",
                "db_id": "absent",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class CorrectButUnverifiableClient:
        def __init__(self, **_):
            pass

        def complete(self, messages, tools):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "final",
                        "final_answer",
                        {"access_granted": True, "sql": "SELECT name FROM people"},
                    )
                ],
            }

    monkeypatch.setattr(
        "src.domains.rbac.rejection_sampling.OpenAICompatibleClient",
        CorrectButUnverifiableClient,
    )
    output = tmp_path / "sampling"
    stats = collect_to_directory(
        config_path=ROOT / "configs/rbac.yaml",
        data_path=data,
        output_dir=output,
        policy_model="fake",
        base_url="http://unused",
        harness_name="initial_harness",
        attempts_per_item=1,
        accepted_per_item=1,
        concurrency=1,
        quality_tier="balanced",
        repair_strategy="verified_only",
        require_sql_execution=True,
        db_dir=str(tmp_path / "missing-databases"),
    )

    attempts = [
        json.loads(line)
        for line in (output / "sampling_attempts.jsonl").read_text().splitlines()
    ]
    assert stats.accepted == 0
    assert attempts[0]["reason"] == "sql_verification_unavailable"


def test_balanced_rbac_repairs_exec_mismatch_with_canonical_gold(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text(
        json.dumps(
            {
                "sample_id": "rbac:wrong-sql",
                "instruction": (
                    "Database: demo\nTotal Tables: 1\n"
                    "Table people has columns such as name.\n"
                    "Role: Analyst\nAccessible Columns: people: name\n\n"
                    "##Access Request:\nlist names"
                ),
                "role": "Analyst",
                "test_role": "Analyst",
                "valid": 1,
                "gold_sql": "SELECT name FROM people ORDER BY name",
                "db_id": "demo",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db_dir = tmp_path / "database"
    (db_dir / "demo").mkdir(parents=True)
    (db_dir / "demo/demo.sqlite").touch()

    class WrongSqlClient:
        def __init__(self, **_):
            pass

        def complete(self, messages, tools):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "wrong-final",
                        "final_answer",
                        {"access_granted": True, "sql": "SELECT name FROM people"},
                    )
                ],
            }

    monkeypatch.setattr(
        "src.domains.rbac.rejection_sampling.OpenAICompatibleClient",
        WrongSqlClient,
    )
    monkeypatch.setattr(
        "src.domains.rbac.rejection_sampling.exec_match_safe",
        lambda *_: False,
    )
    output = tmp_path / "sampling"
    callback_snapshots = []

    def on_attempt(record):
        callback_snapshots.append(
            (
                record,
                (output / "accepted_rollouts.jsonl").read_text(encoding="utf-8"),
            )
        )

    stats = collect_to_directory(
        config_path=ROOT / "configs/rbac.yaml",
        data_path=data,
        output_dir=output,
        policy_model="fake",
        base_url="http://unused",
        harness_name="initial_harness",
        attempts_per_item=2,
        accepted_per_item=1,
        concurrency=2,
        quality_tier="balanced",
        repair_strategy="gold_final",
        db_dir=str(db_dir),
        on_attempt=on_attempt,
    )

    records = [
        json.loads(line) for line in (output / "accepted_rollouts.jsonl").read_text().splitlines()
    ]
    assert (stats.accepted, stats.strict_accepted, stats.repaired_accepted) == (1, 0, 1)
    assert stats.outcome_counts == {"exec_mismatch": 2}
    assert records[0]["sampling"]["quality_tier"] == "balanced_repaired"
    assert records[0]["sampling"]["curation_reason"] == "gold_answer_repair"
    final = records[0]["messages"][-2]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(final) == {
        "access_granted": True,
        "sql": "SELECT name FROM people ORDER BY name",
    }
    assert all(snapshot.strip() for _, snapshot in callback_snapshots)


def test_strict_rbac_does_not_repair_completed_mismatch(tmp_path, monkeypatch):
    data = tmp_path / "train.csv"
    data.write_text(
        "instruction,output,role,test_role,valid\nallow request,answer,Analyst,Analyst,1\n",
        encoding="utf-8",
    )

    class WrongLabelClient:
        def __init__(self, **_):
            pass

        def complete(self, messages, tools):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call("wrong-final", "final_answer", {"access_granted": False})
                ],
            }

    monkeypatch.setattr(
        "src.domains.rbac.rejection_sampling.OpenAICompatibleClient",
        WrongLabelClient,
    )
    stats = collect_to_directory(
        config_path=ROOT / "configs/rbac.yaml",
        data_path=data,
        output_dir=tmp_path / "strict",
        policy_model="fake",
        base_url="http://unused",
        harness_name="initial_harness",
        attempts_per_item=1,
        accepted_per_item=1,
        concurrency=1,
        quality_tier="strict",
    )
    assert (stats.accepted, stats.repaired_accepted) == (0, 0)
