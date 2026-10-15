"""Concrete service wiring without launching vLLM, SFT or MetaAgent processes."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.meta_evolver.services import LocalRuntimeServices


def test_process_service_uses_project_cwd_and_preserves_failure_status(tmp_path):
    services = LocalRuntimeServices(tmp_path, training_python=sys.executable)
    command = [sys.executable, "-c", "from pathlib import Path; Path('marker').write_text('ok')"]
    assert services.run(command, module="unit", dry_run=True) is None
    assert not (tmp_path / "marker").exists()
    assert services.run(command, module="unit", dry_run=False).returncode == 0
    assert (tmp_path / "marker").read_text() == "ok"
    failing = [sys.executable, "-c", "raise SystemExit(7)"]
    with pytest.raises(subprocess.CalledProcessError) as error:
        services.run(failing, module="unit", dry_run=False)
    assert error.value.returncode == 7
    assert services.run(failing, module="unit", dry_run=False, check=False).returncode == 7


def test_serving_service_preserves_start_stop_and_dry_run_contract(tmp_path, monkeypatch):
    from src.meta_evolver import services as module

    events = []
    server = SimpleNamespace(stop=lambda: events.append("stop"))
    monkeypatch.setattr(
        module, "kill_local_vllm_processes", lambda **kw: events.append(("cleanup", kw)) or []
    )
    monkeypatch.setattr(
        module.ManagedVllmServer, "start", lambda **kw: events.append(("start", kw)) or server
    )
    services = LocalRuntimeServices(tmp_path, training_python=sys.executable)
    options = {
        "label": "validation",
        "round_dir": tmp_path / "round",
        "base_url": "http://localhost:8000/v1",
        "expected_model": "policy",
        "settings": {"startup_timeout_seconds": 23},
    }
    assert services.start_vllm(["vllm", "serve", "model"], dry_run=True, **options) is None
    assert not events
    assert services.start_vllm(["vllm", "serve", "model"], dry_run=False, **options) is server
    assert events[0][0] == "cleanup"
    assert events[1][1]["project_root"] == tmp_path
    assert events[1][1]["startup_timeout_seconds"] == 23
    services.stop_vllm(server, label="validation", dry_run=True)
    assert len(events) == 2
    services.stop_vllm(server, label="validation", dry_run=False)
    assert events[-1] == "stop"
