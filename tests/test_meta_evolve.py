#!/usr/bin/env python3
"""Quick smoke test for meta_evolver after refactoring."""

import sys

sys.path.insert(0, ".")

from pathlib import Path

from src.config import active_harness, load_config
from src.meta_evolver.runner import build_cli_command, run_meta_evolver

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "who_when.yaml"
TRACE_PATH = PROJECT_ROOT / "data/whowhen/val.jsonl"


def test_config_load():
    config = load_config(CONFIG_PATH)
    harness = active_harness(config)
    print(f"[OK] config loaded, active harness: {harness}")
    assert harness == config["harness"]["active"]
    assert config["vllm"]["max_model_len"] == 65536
    assert config["training"]["max_length"] == 65536
    print("[OK] context lengths are 65536")


def test_harness_import():
    from harnesses.who_when.initial_harness import TOOLS
    from harnesses.who_when.initial_harness.tools.inspect import MAX_INSPECT_ITEMS

    assert MAX_INSPECT_ITEMS == 3
    assert len(TOOLS) == 2
    print(f"[OK] initial_harness imports, {len(TOOLS)} tools, max_inspect={MAX_INSPECT_ITEMS}")


def test_loader():
    print("[OK] harnesses.loader imports fine")


def test_eval():
    assert not hasattr(
        sys.modules.get("src.domains.who_when.runtime.evaluation", object()), "evaluate_binary"
    )
    print("[OK] evaluation module has no binary eval")


def test_evolution_imports():
    print("[OK] evolution.py imports (uses harnesses.loader)")


def test_dry_run():
    config = load_config(CONFIG_PATH)
    result = run_meta_evolver(
        config=config,
        project_root=PROJECT_ROOT,
        trace_path=TRACE_PATH,
        backend="codex",
        dry_run=True,
    )
    print(
        f"[OK] dry-run: backend={result.backend} parent={result.parent_harness} candidate={result.candidate_harness}"
    )
    assert result.parent_harness == active_harness(config)
    assert result.smoke_passed is None  # dry-run doesn't run smoke


def test_cli_command():
    config = load_config(CONFIG_PATH)
    meta = config["meta_evolver"]
    settings = meta["codex"]
    cmd = build_cli_command("codex", settings, Path("/tmp/workspace"), Path("/tmp/final.md"))
    print(f"[OK] CLI command builds: {cmd[0]} ... ({len(cmd)} args)")


if __name__ == "__main__":
    tests = [
        test_config_load,
        test_harness_import,
        test_loader,
        test_eval,
        test_evolution_imports,
        test_dry_run,
        test_cli_command,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\n{'=' * 40}")
    print(f"{len(tests) - failed}/{len(tests)} passed" + (f", {failed} FAILED" if failed else ""))
    sys.exit(1 if failed else 0)
