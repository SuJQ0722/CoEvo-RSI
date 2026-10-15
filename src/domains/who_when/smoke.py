"""Who&When candidate contract, owned entirely by this domain."""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any

from src.config import harness_module_suffix
from src.domains.base import SmokeCheck, SmokeReport
from src.traces import AgentTrace


def _module_name(package: str, harness_name: str) -> str:
    return (
        f"{package}.initial_harness"
        if harness_name == "system_zero"
        else f"{package}.{harness_module_suffix(harness_name)}"
    )


def _build(module: Any, harness_name: str, trace: AgentTrace) -> Any:
    builder = getattr(module, "build_harness", None)
    if not callable(builder):
        raise TypeError(f"{module.__name__} must expose build_harness(trace)")
    harness = builder(trace)
    required = (
        "tools", "initial_messages", "handle_tool_call", "is_done", "result",
        "inspected_step_ids",
    )
    missing = [name for name in required if not hasattr(harness, name)]
    if missing:
        raise TypeError(f"{module.__name__} harness is missing: {', '.join(missing)}")
    if harness_name not in {"initial_harness", "system_zero"}:
        setattr(harness, "name", harness_name)
    return harness


def _runtime_modules(module: Any, harness: Any) -> set[str]:
    pending: list[Any] = [getattr(module, "build_harness", None)]
    for value in vars(type(harness)).values():
        pending.append(value.fget if isinstance(value, property) else value)
    seen: set[int] = set()
    origins: set[str] = set()
    while pending:
        value = pending.pop()
        if inspect.ismethod(value):
            value = value.__func__
        if not inspect.isfunction(value) or id(value) in seen:
            continue
        seen.add(id(value))
        origin = str(getattr(value, "__module__", ""))
        if origin.startswith("harnesses."):
            origins.add(origin)
        for name in value.__code__.co_names:
            dependency = value.__globals__.get(name)
            if inspect.isfunction(dependency):
                pending.append(dependency)
    return origins


def test(harness_name: str, *, package: str, root: Path) -> SmokeReport:
    checks: list[SmokeCheck] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append(SmokeCheck(name, passed, detail))

    module_name = _module_name(package, harness_name)
    trace = AgentTrace.from_record({
        "sample_id": "SMOKE_PRIVATE_ID",
        "source_dataset": "SMOKE_PRIVATE_SOURCE",
        "task": "Classify this execution.",
        "is_failure": True,
        "trajectory": [
            {"step": 1, "role": "assistant", "content": "SMOKE_SECRET_STEP_TEXT"}
        ],
    })
    try:
        importlib.invalidate_caches()
        module = importlib.import_module(module_name)
        harness = _build(module, harness_name, trace)
    except Exception as exc:
        record("import_and_build", False, f"{type(exc).__name__}: {exc}")
        return SmokeReport(harness_name, False, tuple(checks))

    record("import_and_build", True, f"built {type(harness).__name__}")
    foreign = sorted(
        origin for origin in _runtime_modules(module, harness)
        if origin != module_name and not origin.startswith(module_name + ".")
    )
    record(
        "local_runtime_dependencies",
        not foreign,
        f"runtime modules stay under {module_name}" if not foreign else f"foreign={foreign}",
    )
    identity = getattr(harness, "name", None)
    expected_identity = "initial_harness" if harness_name == "system_zero" else harness_name
    record(
        "identity",
        identity == expected_identity,
        f"expected {expected_identity!r}, got {identity!r}",
    )
    tool_names = [tool.get("function", {}).get("name") for tool in harness.tools]
    record("two_tool_contract", tool_names == ["inspect", "final_answer"], f"tools={tool_names}")

    try:
        serialized = json.dumps(harness.initial_messages(), ensure_ascii=False)
        leaked = [
            marker for marker in (
                "SMOKE_SECRET_STEP_TEXT", "SMOKE_PRIVATE_ID", "SMOKE_PRIVATE_SOURCE",
                '"is_failure"',
            ) if marker in serialized
        ]
        record("summary_boundary", not leaked, "no private fields leaked" if not leaked else f"leaked={leaked}")
    except Exception as exc:
        record("summary_boundary", False, f"{type(exc).__name__}: {exc}")

    try:
        inspected = json.loads(harness.handle_tool_call("inspect", {"step_id_list": [1]}))
        ok = (
            inspected.get("steps", [{}])[0].get("record", {}).get("content")
            == "SMOKE_SECRET_STEP_TEXT"
            and harness.inspected_step_ids == [1]
        )
        record("inspect_execution", ok, "selected step returned" if ok else str(inspected))
    except Exception as exc:
        record("inspect_execution", False, f"{type(exc).__name__}: {exc}")

    try:
        harness.handle_tool_call("final_answer", {
            "is_failure": True,
            "evidence_step_ids": [1],
            "responsible_agent": "assistant",
        })
        ok = bool(harness.is_done and harness.result and harness.result.is_failure)
        record("final_answer_execution", ok, "terminal result stored")
    except Exception as exc:
        record("final_answer_execution", False, f"{type(exc).__name__}: {exc}")

    return SmokeReport(harness_name, all(check.passed for check in checks), tuple(checks))
