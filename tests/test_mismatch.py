"""Offline mismatch diagnostics and unchanged model execution contracts."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.diagnostics.mismatch import (
    MismatchContract,
    analyze_record,
    round_comparisons,
    summarize,
    write_report,
)
from src.diagnostics.observe import ObservedModel
from src.domains.homebench.mismatch import HomeBenchMismatch
from src.domains.rbac.mismatch import RBACMismatch
from src.domains.who_when.mismatch import WhoWhenMismatch

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_id_list": {"type": "array", "items": {"type": "integer"}, "minItems": 1}
                },
                "required": ["step_id_list"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "parameters": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        },
    },
]


def call(name="inspect", args=None, cid="one"):
    return {
        "id": cid,
        "type": "function",
        "function": {
            "name": name,
            "arguments": {"step_id_list": [1]} if args is None else args,
        },
    }


def record(*responses, finished=True):
    return {
        "sample_id": "sample",
        "prediction": {"is_failure": True} if finished else None,
        "interface_trace": {
            "version": 1,
            "requests": [
                {"tools": deepcopy(TOOLS), "response": {"role": "assistant", "tool_calls": r}}
                for r in responses
            ],
        },
        "messages": [],
    }


def report(row, contract=None):
    return summarize([analyze_record(row, contract or WhoWhenMismatch())])


@pytest.mark.parametrize(
    "name,args,code",
    [
        ("missing", {}, "unknown_tool"),
        ("inspect", "{", "invalid_json"),
        ("inspect", "[]", "not_object"),
        ("inspect", {}, "missing_required"),
        ("inspect", {"step_id_list": "1"}, "wrong_type"),
        ("inspect", {"step_id_list": [1], "extra": 1}, "extra_property"),
        ("inspect", {"step_id_list": []}, "enum_or_bound"),
    ],
)
def test_classifies_calls_once(name, args, code):
    row = analyze_record(record([call(name, args)]), WhoWhenMismatch())
    assert code in {e["code"] for e in row["events"]}
    result = summarize([row])
    assert result["rates"]["interface_mismatch_task_rate"]["rate"] == 1
    key = "tool_name_error_rate" if name == "missing" else "tool_argument_error_rate"
    assert result["rates"][key]["count"] == 1
    if name == "missing":
        assert result["rates"]["tool_argument_error_rate"]["denominator"] == 0


def test_task_dedup_and_recovery_are_separate_from_task_accuracy():
    row = record([call(args="{")], [call("final_answer", {"ok": True}, "two")])
    result = summarize([analyze_record(row, WhoWhenMismatch())])
    assert result["rates"]["interface_mismatch_task_rate"]["count"] == 1
    assert result["rates"]["interface_recovery_rate"]["rate"] == 1
    assert result["recovered_task_quality"] == {"unknown": 1}
    assert result["rates"]["tool_return_error_rate"]["rate"] is None


def test_turn_imr_unions_three_error_categories_without_double_counting():
    class FinalFormatBad(WhoWhenMismatch):
        def state(self, record):
            return {"finished": False, "final_format_error": True, "rule_only": False}

    row = record(
        [call(args="{"), call(cid="two")],
        [call("final_answer", {"ok": True}, "three")],
    )
    result = report(row, FinalFormatBad())

    assert result["model_response_count"] == 2
    assert result["confirmed_mismatch_turn_count"] == 2
    assert result["rates"]["protocol_mismatch_turn_rate"] == {
        "count": 1,
        "denominator": 2,
        "rate": 0.5,
        "reason": None,
    }
    assert result["rates"]["argument_mismatch_turn_rate"]["rate"] == 0.5
    assert result["rates"]["final_format_mismatch_turn_rate"]["rate"] == 0.5
    # Turn zero has both a protocol and an argument error, but the IMR union
    # still counts it only once.
    assert result["rates"]["interface_mismatch_turn_rate"]["count"] == 2
    assert result["rates"]["interface_mismatch_turn_rate"]["rate"] == 1.0
    assert result["coverage"]["complete_interface_turns"]["rate"] == 1.0


def test_invalid_final_then_success_and_terminal_without_tool_return():
    result = report(
        record([call("final_answer", "{")], [call("final_answer", {"ok": True}, "two")])
    )
    assert result["rates"]["final_format_error_rate"]["rate"] == 0
    assert result["rates"]["ever_final_format_error_rate"]["rate"] == 1
    assert result["rates"]["interface_recovery_rate"]["rate"] == 1
    assert result["tool_return_count"] == 0


def test_protocol_limits_follow_domain_and_id_repairs_are_observed():
    multiple = record([call(cid="one"), call(cid="two")])
    assert report(multiple)["rates"]["response_protocol_error_rate"]["rate"] == 1
    assert report(multiple, RBACMismatch())["rates"]["response_protocol_error_rate"]["rate"] == 0
    for calls in ([], [call(cid=None)], [call(), call()]):
        assert report(record(calls))["rates"]["response_protocol_error_rate"]["rate"] == 1


def test_successful_transport_repair_uses_effective_response_for_imr():
    row = record([])
    request = row["interface_trace"]["requests"][0]
    request["transport_repaired"] = True
    request["effective_response"] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [call()],
    }
    result = report(row)

    assert result["rates"]["response_protocol_error_rate"]["rate"] == 0
    assert result["rates"]["interface_mismatch_turn_rate"]["rate"] == 0
    assert result["rates"]["transport_repair_rate"]["rate"] == 1


def test_repaired_response_still_counts_if_effective_protocol_is_invalid():
    row = record([])
    request = row["interface_trace"]["requests"][0]
    request["transport_repaired"] = True
    request["effective_response"] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [call(cid="one"), call(cid="two")],
    }
    result = report(row)

    assert result["rates"]["response_protocol_error_rate"]["rate"] == 1
    assert result["rates"]["interface_mismatch_turn_rate"]["rate"] == 1
    assert result["rates"]["transport_repair_rate"]["rate"] == 1


def test_transport_owned_recovered_ids_are_not_model_mismatch():
    row = record([], [])
    for request in row["interface_trace"]["requests"]:
        request["transport_repaired"] = True
        request["effective_response"] = {
            "role": "assistant",
            "content": "",
            "tool_calls": [call(cid="legacy-repeated-recovery-id")],
        }

    result = report(row)

    assert result["rates"]["response_protocol_error_rate"]["rate"] == 0
    assert result["rates"]["interface_mismatch_turn_rate"]["rate"] == 0


def test_legacy_successful_transport_repair_does_not_use_raw_shape_as_imr():
    row = record([])
    row["interface_trace"]["requests"][0]["transport_repaired"] = True

    result = report(row)

    assert result["rates"]["response_protocol_error_rate"]["rate"] == 0
    assert result["rates"]["interface_mismatch_turn_rate"]["rate"] == 0


def test_observer_detaches_schema_and_raw_response_and_calls_once():
    class Model:
        def complete(self, messages, tools):
            return {"tool_calls": [call(cid=None)]}

    model = ObservedModel(Model())
    tools = deepcopy(TOOLS)
    response = model.complete([], tools)
    response["tool_calls"][0]["id"] = "repaired"
    tools[0]["function"]["name"] = "changed"
    trace = model.trace()
    assert len(trace["requests"]) == 1
    assert trace["requests"][0]["response"]["tool_calls"][0]["id"] is None
    assert trace["requests"][0]["tools"][0]["function"]["name"] == "inspect"


def test_provider_exception_passes_through_and_is_not_model_protocol_failure():
    class Model:
        def complete(self, messages, tools):
            raise TimeoutError("provider unavailable")

    model = ObservedModel(Model())
    with pytest.raises(TimeoutError):
        model.complete([], TOOLS)
    result = report({"sample_id": "x", "interface_trace": model.trace("provider_error")})
    assert result["provider_error_task_count"] == 1
    assert result["rates"]["interface_mismatch_task_rate"]["rate"] is None


def test_rule_only_tasks_and_legacy_coverage_never_look_perfect():
    rule = {
        "sample_id": "rule",
        "rule_used": 1,
        "llm_called": 0,
        "interface_trace": {"version": 1, "requests": []},
    }
    result = report(rule, RBACMismatch())
    assert result["rule_only_task_count"] == 1
    assert result["rates"]["interface_mismatch_task_rate"]["rate"] is None
    legacy = {
        "sample_id": "old",
        "messages": [{"role": "assistant", "tool_calls": [call(args="{")]}],
    }
    result = report(legacy)
    assert result["rates"]["tool_name_error_rate"]["rate"] is None
    assert result["rates"]["argument_json_error_rate"]["rate"] == 1
    assert result["rates"]["interface_mismatch_task_rate"]["rate"] is None
    assert result["confirmed_mismatch_task_count"] == 1


@pytest.mark.parametrize(
    "schema", [{"type": "invalid"}, {"$ref": "https://must-not-fetch.invalid/schema"}]
)
def test_bad_harness_schema_is_unknown_and_never_fetched(schema):
    row = record([call()])
    row["interface_trace"]["requests"][0]["tools"][0]["function"]["parameters"] = schema
    result = report(row)
    assert result["rates"]["tool_argument_error_rate"]["rate"] is None
    assert result["rates"]["interface_mismatch_task_rate"]["rate"] is None
    assert result["confirmed_mismatch_task_count"] == 0


def test_tool_errors_use_envelopes_not_embedded_error_text():
    ww = WhoWhenMismatch()
    assert (
        ww.tool_return_errors("inspect", {"steps": [{"record": {"error": "original failure"}}]})
        == []
    )
    assert ww.tool_return_errors("inspect", {"steps": [{"error": "step not found"}]}) == [
        "argument_semantic"
    ]
    rb = RBACMismatch()
    assert rb.tool_return_errors(
        "test_sql", {"error": "no database available for this sample"}
    ) == ["infrastructure"]
    assert rb.tool_return_errors("test_sql", {"error": "OperationalError: bad SQL"}) == [
        "execution"
    ]
    assert (
        rb.tool_return_errors("final_answer", {"status": "submitted", "result": "ACCESS_DENIED"})
        == []
    )
    hb = HomeBenchMismatch()
    assert hb.tool_return_errors("inspect_home", {"room": {"error": "unknown room"}}) == [
        "argument_semantic"
    ]
    assert (
        hb.tool_return_errors("inspect_home", {"room": {"device": {"error": "stored data"}}}) == []
    )
    assert hb.tool_return_errors("list_methods", []) == []
    assert hb.tool_return_errors("list_methods", "malformed") == ["return_protocol"]


def test_infrastructure_tool_error_not_in_mismatch_union():
    row = record([call()])
    row["messages"] = [
        {
            "role": "tool",
            "name": "test_sql",
            "tool_call_id": "one",
            "content": '{"error":"no database available for this sample"}',
        }
    ]
    result = report(row, RBACMismatch())
    assert result["rates"]["tool_return_error_rate"]["rate"] == 1
    assert result["rates"]["interface_mismatch_task_rate"]["rate"] == 0
    assert result["tool_return_categories"] == {"infrastructure": 1}


def test_return_semantic_error_then_final_recovers():
    row = record([call()], [call("final_answer", {"ok": True}, "two")])
    row["messages"] = [
        {
            "role": "tool",
            "name": "inspect",
            "tool_call_id": "one",
            "content": '{"steps":[{"error":"step not found"}]}',
        }
    ]
    result = report(row)
    assert result["rates"]["interface_recovery_rate"]["rate"] == 1


def test_homebench_business_error_and_transport_are_valid():
    row = {
        "finished": True,
        "generated": "error_input",
        "messages": [
            {"role": "assistant", "tool_calls": [call("final_answer", {"commands": "error_input"})]}
        ],
    }
    assert HomeBenchMismatch().state(row)["final_format_error"] is False
    row["generated"] = "{broken"
    row["messages"][0]["tool_calls"][0]["function"]["arguments"]["commands"] = "{broken"
    assert HomeBenchMismatch().state(row)["final_format_error"] is True


def test_reports_are_deterministic_and_backfill_is_offline(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(record([call()])) + "\n")
    result = write_report(path, WhoWhenMismatch(), identity={"pair_id": "pair"})
    before = Path(result["report_path"]).read_bytes(), Path(result["samples_path"]).read_bytes()
    write_report(path, WhoWhenMismatch(), identity={"pair_id": "pair"})
    assert before == (
        Path(result["report_path"]).read_bytes(),
        Path(result["samples_path"]).read_bytes(),
    )


def test_unknown_domain_has_no_invented_terminal_or_schema_checks():
    result = report({"sample_id": "empty"}, MismatchContract())
    assert all(rate["rate"] is None for rate in result["rates"].values())


def test_training_comparison_requires_full_coverage():
    before = report(record([call(args="{")]))
    after = report(record([call("final_answer", {"ok": True})]))
    pre = {
        "id": "pre",
        "harness": "h",
        "model": {"id": "m"},
        "scores": {"q": 1},
        "split": "val",
        "diagnostics": {"mismatch": before},
    }
    post = {**pre, "id": "post", "diagnostics": {"mismatch": after}}
    journal = {
        "evaluations": {"pre": pre, "post": post},
        "training": {"one": {"sampling_pair_id": "pre", "post": post}},
    }
    change = round_comparisons(journal)["training_changes"][0]
    assert change["rate_deltas_post_minus_pre"]["interface_mismatch_task_rate"] == -1
    after["coverage"]["complete_interface_turns"]["rate"] = None
    assert not round_comparisons(journal)["training_changes"][0]["comparable"]


def test_round_comparison_labels_selected_and_latest_trained_imr():
    before = report(record([call(args="{")]))
    after = report(record([call("final_answer", {"ok": True})]))
    for mismatch in (before, after):
        mismatch["identity"] = {"tasks": {"sha256": "same-val"}}
    selected = {
        "id": "selected",
        "harness": "h",
        "model": {"id": "base"},
        "scores": {"q": 1.0},
        "phase": "pre",
        "split": "val",
        "diagnostics": {"mismatch": before},
    }
    latest = {
        **selected,
        "id": "latest",
        "model": {"id": "trained"},
        "scores": {"q": 2.0},
        "phase": "post",
        "diagnostics": {"mismatch": after},
    }
    journal = {
        "evaluations": {"selected": selected, "latest": latest},
        "training": {"1": {"sampling_pair_id": "selected", "post": latest}},
    }

    comparison = round_comparisons(
        journal, selected_pair_id="selected"
    )["selected_vs_latest_trained"]

    assert comparison["comparable"] is True
    assert comparison["selected"]["model_id"] == "base"
    assert comparison["latest_trained"]["model_id"] == "trained"
    assert comparison["imr_delta_latest_minus_selected"] == -1
    assert comparison["quality_deltas_latest_minus_selected"] == {"q": 1.0}


def test_unknown_tool_error_is_not_assumed_to_be_a_model_mistake():
    contract = WhoWhenMismatch()
    assert contract.tool_return_errors("inspect", {"error": "unexpected failure"}) == ["unknown"]
    assert contract.tool_return_errors("inspect", {"steps": [{"error": "unexpected failure"}]}) == [
        "unknown"
    ]


def test_offline_cli_uses_existing_rollouts(tmp_path):
    import subprocess
    import sys

    source = tmp_path / "rollouts.jsonl"
    source.write_text(json.dumps(record([call(args="{")])) + "\n")
    original = source.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.diagnostics",
            "--domain",
            "who_when",
            "--rollouts",
            str(source),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["rates"]["tool_argument_error_rate"]["rate"] == 1
    assert source.read_bytes() == original
    assert (tmp_path / "mismatch_samples.jsonl").exists()
