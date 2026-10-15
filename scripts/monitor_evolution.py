#!/usr/bin/env python3
"""Monitor an CoEvoX command and ask Codex to repair code-only failures."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_output_root  # noqa: E402
from src.meta_evolver.runner import (  # noqa: E402
    _subprocess_environment,
    build_cli_command,
)
from src.utils.persistence import atomic_json

MEMORY_PATTERNS = (
    re.compile(r"cuda out of memory", re.I),
    re.compile(r"torch\.outofmemoryerror", re.I),
    re.compile(r"cublas_status_alloc_failed", re.I),
    re.compile(r"hip out of memory", re.I),
    re.compile(r"out of memory.*gpu", re.I),
    re.compile(r"insufficient free gpu memory", re.I),
)
ENVIRONMENT_PATTERNS = (
    re.compile(r"cuda driver.*(?:insufficient|not found|failed)", re.I),
    re.compile(r"cuda initialization error", re.I),
    re.compile(r"no cuda gpus? are available", re.I),
    re.compile(r"nccl.*(?:system error|unhandled|internal error)", re.I),
    re.compile(r"no space left on device", re.I),
    re.compile(r"read-only file system", re.I),
    re.compile(r"permission denied.*(?:cuda|model|checkpoint|workspace)", re.I),
    re.compile(r"vllm executable (?:is missing|not found)", re.I),
    re.compile(r"python environment is missing", re.I),
    re.compile(r"model endpoint is not ready", re.I),
    re.compile(r"connection (?:refused|timed out).*(?:api|endpoint|server)", re.I),
)
CLASSIFICATION = re.compile(
    r"^CLASSIFICATION:\s*(code|memory|environment|unknown)\s*$",
    re.I | re.M,
)
REPAIR_STATUS = re.compile(r"^REPAIR_STATUS:\s*(fixed|not_fixed)\s*$", re.I | re.M)


def deterministic_failure_classification(text: str) -> str | None:
    if any(pattern.search(text) for pattern in MEMORY_PATTERNS):
        return "memory"
    if any(pattern.search(text) for pattern in ENVIRONMENT_PATTERNS):
        return "environment"
    return None


def _run_and_tee(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return process.wait()


def _recent_logs(since: float, primary: Path, output_root: Path, limit: int = 10) -> list[Path]:
    candidates = [primary]
    if output_root.is_dir():
        candidates.extend(
            path
            for path in output_root.rglob("*.log")
            if path.is_file() and path.stat().st_mtime >= since - 1
        )
    unique = {path.resolve(): path.resolve() for path in candidates}
    return sorted(unique.values(), key=lambda path: path.stat().st_mtime, reverse=True)[:limit]


def _codex_command(
    *,
    settings: dict[str, Any],
    workspace: Path,
    final_message: Path,
    read_only: bool,
) -> list[str]:
    effective = dict(settings)
    effective["multi_agent"] = False
    command = build_cli_command("codex", effective, workspace, final_message)
    if read_only:
        index = command.index("--sandbox") + 1
        command[index] = "read-only"
    return command


def _call_codex(
    *,
    settings: dict[str, Any],
    monitor_root: Path,
    name: str,
    prompt: str,
    read_only: bool,
) -> tuple[int, str]:
    final_message = monitor_root / f"{name}.md"
    command = _codex_command(
        settings=settings,
        workspace=PROJECT_ROOT,
        final_message=final_message,
        read_only=read_only,
    )
    codex_home = monitor_root / f"{name}_codex_home"
    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        cwd=PROJECT_ROOT,
        env=_subprocess_environment(settings, codex_home, PROJECT_ROOT),
        check=False,
    )
    (monitor_root / f"{name}.stdout.jsonl").write_text(completed.stdout, encoding="utf-8")
    (monitor_root / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    text = (
        final_message.read_text(encoding="utf-8", errors="replace")
        if final_message.is_file()
        else completed.stdout
    )
    return completed.returncode, text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-code-repairs", type=int, default=3)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a monitored command is required after --")
    if args.max_code_repairs < 0:
        parser.error("--max-code-repairs must be non-negative")

    config = load_config(args.config.resolve())
    output_root = resolve_output_root(config, PROJECT_ROOT)
    meta = dict(config.get("meta_evolver") or {})
    settings = dict(meta.get("codex") or {})
    if not settings:
        parser.error("meta_evolver.codex configuration is required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    monitor_root = output_root / "monitor" / stamp
    monitor_root.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "command": command,
        "config": str(args.config),
        "attempts": [],
        "status": "running",
    }
    state_path = monitor_root / "status.json"
    atomic_json(state_path, state)

    for attempt in range(1, args.max_code_repairs + 2):
        started = time.time()
        attempt_log = monitor_root / f"attempt_{attempt:02d}.log"
        print(f"monitor attempt={attempt} log={attempt_log}", flush=True)
        return_code = _run_and_tee(command, attempt_log)
        entry: dict[str, Any] = {
            "attempt": attempt,
            "return_code": return_code,
            "log": str(attempt_log),
        }
        state["attempts"].append(entry)
        if return_code == 0:
            state["status"] = "completed"
            atomic_json(state_path, state)
            print(f"monitor complete status=completed attempts={attempt}", flush=True)
            return

        logs = _recent_logs(started, attempt_log, output_root)
        combined = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")[-200_000:] for path in logs
        )
        deterministic = deterministic_failure_classification(combined)
        log_list = "\n".join(f"- {path}" for path in logs)
        diagnosis_prompt = f"""You are the read-only failure classifier for an CoEvoX
evolution run. Read the listed latest logs; do not modify any file and do
not install dependencies. Classify the primary root cause as exactly one of:
code, memory, environment, unknown.

Use `memory` for GPU/CPU OOM or allocation exhaustion. Use `environment` for
CUDA/driver/NCCL faults, missing dependencies or executables, permissions,
disk/network/service failures, and corrupt/missing external model artifacts.
Use `code` only when repository source logic is demonstrably wrong and can be
fixed without changing data, credentials, hardware allocation, dependencies,
or external services. Otherwise use `unknown`.

Logs:
{log_list}

Your first final line must be `CLASSIFICATION: <value>`, followed by concise
evidence with exact file/traceback locations. Never make edits in this step."""
        diagnostic_rc, diagnosis = _call_codex(
            settings=settings,
            monitor_root=monitor_root,
            name=f"attempt_{attempt:02d}_diagnosis",
            prompt=diagnosis_prompt,
            read_only=True,
        )
        match = CLASSIFICATION.search(diagnosis)
        classification = match.group(1).lower() if match else "unknown"
        if deterministic in {"memory", "environment"}:
            classification = deterministic
        entry.update(
            {
                "classification": classification,
                "diagnostic_return_code": diagnostic_rc,
                "diagnosis": str(monitor_root / f"attempt_{attempt:02d}_diagnosis.md"),
                "related_logs": [str(path) for path in logs],
            }
        )
        atomic_json(state_path, state)
        if classification != "code":
            state["status"] = f"stopped_{classification}"
            atomic_json(state_path, state)
            print(
                f"monitor stop classification={classification} diagnosis={entry['diagnosis']}",
                flush=True,
            )
            raise SystemExit(return_code or 1)
        if attempt > args.max_code_repairs:
            state["status"] = "stopped_repair_limit"
            atomic_json(state_path, state)
            print("monitor stop: code repair limit exhausted", flush=True)
            raise SystemExit(return_code or 1)

        repair_prompt = f"""You are the bounded code-repair worker for a failed
CoEvoX evolution run. Read the diagnosis and logs below, reproduce the
repository code failure with a focused test, make the minimal source-code fix,
and run bounded relevant tests.

Diagnosis: {entry["diagnosis"]}
Logs:
{log_list}

Only repair repository code. Preserve unrelated dirty-worktree changes. Do not
change `.env`, credentials, datasets, operator/gold files, generated rollout
outputs, model/checkpoint files, CUDA allocation, dependency versions, or system
environment. Do not install packages or start a training/evaluation run. If the
problem is actually memory/environment/external/unknown, make no edits.

Your final response must begin with `REPAIR_STATUS: fixed` only if a focused
test proves the code fix; otherwise begin with `REPAIR_STATUS: not_fixed`. List
changed files and tests."""
        repair_rc, repair = _call_codex(
            settings=settings,
            monitor_root=monitor_root,
            name=f"attempt_{attempt:02d}_repair",
            prompt=repair_prompt,
            read_only=False,
        )
        repair_match = REPAIR_STATUS.search(repair)
        repair_status = repair_match.group(1).lower() if repair_match else "not_fixed"
        entry.update(
            {
                "repair_return_code": repair_rc,
                "repair_status": repair_status,
                "repair_report": str(monitor_root / f"attempt_{attempt:02d}_repair.md"),
            }
        )
        atomic_json(state_path, state)
        if repair_rc != 0 or repair_status != "fixed":
            state["status"] = "stopped_code_not_fixed"
            atomic_json(state_path, state)
            print(f"monitor stop repair_status={repair_status}", flush=True)
            raise SystemExit(return_code or 1)
        print("monitor restarting after verified code repair", flush=True)


if __name__ == "__main__":
    main()
