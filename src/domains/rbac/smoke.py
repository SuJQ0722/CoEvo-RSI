"""Candidate contract checks for RBAC harness evolution."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from src.config import harness_module_suffix
from src.domains.base import SmokeCheck, SmokeReport


def test(harness_name: str, *, package: str, root: Path) -> SmokeReport:
    checks: list[SmokeCheck] = []
    module_name = (
        f"{package}.initial_harness"
        if harness_name == "system_zero"
        else f"{package}.{harness_module_suffix(harness_name)}"
    )
    policy = [
        {"instruction": "CEO secret", "role": "CEO (t)", "test_role": "CEO (t)", "valid": 1},
        {
            "instruction": "Team note",
            "role": "CEO (t) - IT (s)",
            "test_role": "CEO (t) - IT (s)",
            "valid": 1,
        },
    ]
    try:
        importlib.invalidate_caches()
        module = importlib.import_module(module_name)
        harness = module.build_harness(policy)
        checks.append(SmokeCheck("import_and_build", True, module_name))
    except Exception as exc:
        checks.append(SmokeCheck("import_and_build", False, f"{type(exc).__name__}: {exc}"))
        return SmokeReport(harness_name, False, tuple(checks))

    invalid = harness.route("", "")
    mismatch = harness.route("Role: Auditor\nAccessible Columns: t: id", "Analyst")
    full = harness.route(
        "Total Tables: 1\nRole: Analyst\nAccessible Columns: t: ALL COLUMNS",
        "Analyst",
    )
    fallback = harness.route(
        "Total Tables: 1\nRole: NewRole\nAccessible Columns: t: id",
        "NewRole",
    )
    checks.append(SmokeCheck("invalid_role_fail_closed", invalid.decision == 0, str(invalid)))
    checks.append(SmokeCheck("role_policy_mismatch", mismatch.decision == 0, str(mismatch)))
    checks.append(SmokeCheck("all_columns_policy", full.decision == 1, str(full)))
    checks.append(SmokeCheck("ambiguous_fallback", fallback.decision is None, str(fallback)))
    bits = [
        harness.parse_access("ACCESS_DENIED"),
        harness.parse_access("ACCESS_GRANTED"),
        harness.parse_access("free-form answer"),
    ]
    checks.append(SmokeCheck("formal_marker_parser", bits == [0, 1, 0], f"bits={bits}"))
    tool_names = [tool.get("function", {}).get("name") for tool in harness.tools]
    tool_map = {
        tool["function"]["name"]: tool["function"]
        for tool in harness.tools
        if "function" in tool
    }
    fa = tool_map.get("final_answer", {})
    fa_params = fa.get("parameters", {})
    fa_props = fa_params.get("properties", {})
    formal_tool = (
        "final_answer" in tool_names
        and fa_params.get("type") == "object"
        and "access_granted" in fa_props
        and fa_props["access_granted"].get("type") == "boolean"
        and "access_granted" in (fa_params.get("required") or [])
        and fa_params.get("additionalProperties") is False
        and all(name in {"final_answer", "test_sql"} for name in tool_names)
    )
    checks.append(
        SmokeCheck(
            "single_final_answer_tool",
            formal_tool,
            f"tools={harness.tools}",
        )
    )

    class FakeModel:
        @staticmethod
        def complete(messages, tools):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "final_answer",
                        "arguments": json.dumps({"access_granted": True}),
                    },
                }],
                "_coevox_usage": {"total_tokens": 3},
            }

    call = harness.call_llm(FakeModel(), "new semantic request", "NewRole", fallback)
    checks.append(
        SmokeCheck(
            "llm_component",
            call.response == "ACCESS_GRANTED"
            and call.usage.get("total_tokens") == 3
            and not call.error,
            "harness owns structured invocation and usage normalization",
        )
    )
    formal_transcript = (
        len(call.messages) >= 2
        and call.messages[-2].get("role") == "assistant"
        and len(call.messages[-2].get("tool_calls") or []) == 1
        and call.messages[-2]["tool_calls"][0].get("function", {}).get("name")
        == "final_answer"
        and call.messages[-1].get("role") == "tool"
    )
    checks.append(
        SmokeCheck(
            "terminal_tool_transcript",
            formal_transcript,
            "assistant final_answer call is followed by its tool result",
        )
    )
    return SmokeReport(harness_name, all(item.passed for item in checks), tuple(checks))
