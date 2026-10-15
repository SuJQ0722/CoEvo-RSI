import sys

import pytest

from src.runner import vllm_server
from src.runner.vllm_server import ManagedVllmServer, build_vllm_command


def test_build_vllm_command_for_base_model():
    command = build_vllm_command(
        settings={
            "command": "vllm",
            "host": "127.0.0.1",
            "port": 9000,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.7,
            "tensor_parallel_size": 2,
            "data_parallel_size": 4,
            "dtype": "bfloat16",
            "extra_args": ["--enforce-eager"],
        },
        model="models/qwen",
        served_model_name="Qwen3.5-4B",
    )
    assert command == [
        "vllm",
        "serve",
        "models/qwen",
        "--host",
        "127.0.0.1",
        "--port",
        "9000",
        "--served-model-name",
        "Qwen3.5-4B",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.7",
        "--tensor-parallel-size",
        "2",
        "--data-parallel-size",
        "4",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
    ]


def test_managed_server_refuses_to_replace_existing_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(vllm_server, "endpoint_is_ready", lambda *args: True)
    with pytest.raises(RuntimeError, match="refusing to replace"):
        ManagedVllmServer.start(
            command=[sys.executable],
            project_root=tmp_path,
            log_path=tmp_path / "server.log",
            pid_path=tmp_path / "server.pid",
            base_url="http://127.0.0.1:8000/v1",
            expected_model="model",
            startup_timeout_seconds=1,
            shutdown_timeout_seconds=1,
        )


def test_managed_server_starts_health_checks_and_stops(tmp_path, monkeypatch):
    class FakeProcess:
        pid = 12345

        def __init__(self):
            self.returncode = None
            self.wait_timeout = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.wait_timeout = timeout
            self.returncode = 0
            return self.returncode

    process = FakeProcess()
    popen_calls = []

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return process

    health_checks = []

    def fake_endpoint_is_ready(base_url, expected_model=None):
        health_checks.append((base_url, expected_model))
        # The first check protects an existing endpoint. The second check is
        # the post-start model-aware readiness probe.
        return expected_model == "model"

    stopped = []
    monkeypatch.setattr(vllm_server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(vllm_server, "endpoint_is_ready", fake_endpoint_is_ready)
    monkeypatch.setattr(
        vllm_server,
        "stop_process_group",
        lambda pgid, timeout: stopped.append((pgid, timeout)),
    )

    pid_path = tmp_path / "server.pid"
    server = ManagedVllmServer.start(
        command=[sys.executable, "-m", "vllm.entrypoints.openai.api_server"],
        project_root=tmp_path,
        log_path=tmp_path / "server.log",
        pid_path=pid_path,
        base_url="http://127.0.0.1:8000/v1",
        expected_model="model",
        startup_timeout_seconds=5,
        shutdown_timeout_seconds=2,
    )
    assert health_checks == [
        ("http://127.0.0.1:8000/v1", None),
        ("http://127.0.0.1:8000/v1", "model"),
    ]
    assert popen_calls[0][1]["start_new_session"] is True
    assert popen_calls[0][1]["env"]["HF_HUB_OFFLINE"] == "1"
    assert popen_calls[0][1]["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert popen_calls[0][1]["env"]["HF_DATASETS_OFFLINE"] == "1"
    assert pid_path.is_file()
    assert pid_path.read_text(encoding="utf-8") == "12345\n"

    server.stop()

    assert stopped == [(12345, 2)]
    assert server.process.poll() is not None
    assert process.wait_timeout == 2
    assert not pid_path.exists()


@pytest.mark.parametrize("leader_alive", [False, True])
def test_managed_server_retries_address_in_use_startup(tmp_path, monkeypatch, leader_alive):
    class FakeProcess:
        def __init__(self, pid, returncode):
            self.pid = pid
            self.returncode = returncode

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 0
            return self.returncode

    processes = [FakeProcess(101, None if leader_alive else 1), FakeProcess(102, None)]
    popen_calls = []

    def fake_popen(command, **kwargs):
        process = processes[len(popen_calls)]
        if not popen_calls:
            kwargs["stdout"].write(
                "torch.distributed.DistNetworkError: EADDRINUSE: address already in use\n"
                "RuntimeError: Engine core initialization failed\n"
            )
            kwargs["stdout"].flush()
        popen_calls.append((command, kwargs))
        return process

    health_checks = 0

    def fake_endpoint_is_ready(*args, **kwargs):
        nonlocal health_checks
        health_checks += 1
        return health_checks == 2

    stopped = []
    monkeypatch.setattr(vllm_server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(vllm_server, "endpoint_is_ready", fake_endpoint_is_ready)
    monkeypatch.setattr(vllm_server.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        vllm_server,
        "stop_process_group",
        lambda pgid, timeout: stopped.append((pgid, timeout)),
    )

    pid_path = tmp_path / "server.pid"
    log_path = tmp_path / "server.log"
    server = ManagedVllmServer.start(
        command=[sys.executable],
        project_root=tmp_path,
        log_path=log_path,
        pid_path=pid_path,
        base_url="http://127.0.0.1:8000/v1",
        expected_model="model",
        startup_timeout_seconds=5,
        shutdown_timeout_seconds=2,
    )

    assert len(popen_calls) == 2
    assert stopped == [(101, 2)]
    assert pid_path.read_text(encoding="utf-8") == "102\n"
    assert "retrying vLLM startup" in log_path.read_text(encoding="utf-8")

    server.stop()


def test_managed_server_does_not_retry_other_startup_errors(tmp_path, monkeypatch):
    class FakeProcess:
        pid = 101
        returncode = 1

        def poll(self):
            return self.returncode

    popen_calls = []

    def fake_popen(command, **kwargs):
        kwargs["stdout"].write("RuntimeError: CUDA out of memory\n")
        kwargs["stdout"].flush()
        popen_calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(vllm_server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(vllm_server, "endpoint_is_ready", lambda *args, **kwargs: False)
    monkeypatch.setattr(vllm_server, "stop_process_group", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match=r"after 1 attempt\(s\)"):
        ManagedVllmServer.start(
            command=[sys.executable],
            project_root=tmp_path,
            log_path=tmp_path / "server.log",
            pid_path=tmp_path / "server.pid",
            base_url="http://127.0.0.1:8000/v1",
            expected_model="model",
            startup_timeout_seconds=5,
            shutdown_timeout_seconds=2,
        )

    assert len(popen_calls) == 1


def test_managed_server_stops_hung_process_after_fatal_log(tmp_path, monkeypatch):
    class FakeProcess:
        pid = 101

        def __init__(self):
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 0
            return self.returncode

    process = FakeProcess()

    def fake_popen(command, **kwargs):
        kwargs["stdout"].write(
            "RuntimeError: DP Coordinator process failed to report ZMQ addresses "
            "during startup.\n"
        )
        kwargs["stdout"].flush()
        return process

    stopped = []
    monkeypatch.setattr(vllm_server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(vllm_server, "endpoint_is_ready", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        vllm_server,
        "stop_process_group",
        lambda pgid, timeout: stopped.append((pgid, timeout)),
    )

    with pytest.raises(RuntimeError, match="fatal startup error.*DP Coordinator"):
        ManagedVllmServer.start(
            command=[sys.executable],
            project_root=tmp_path,
            log_path=tmp_path / "server.log",
            pid_path=tmp_path / "server.pid",
            base_url="http://127.0.0.1:8000/v1",
            expected_model="model",
            startup_timeout_seconds=60,
            shutdown_timeout_seconds=2,
        )

    assert stopped == [(101, 2)]


def test_managed_server_does_not_reuse_prior_attempt_port_error(
    tmp_path, monkeypatch
):
    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid
            self.returncode = 1

        def poll(self):
            return self.returncode

    processes = [FakeProcess(101), FakeProcess(102)]
    errors = [
        "EADDRINUSE: address already in use\n",
        "RuntimeError: CUDA out of memory\n",
    ]
    popen_calls = []

    def fake_popen(command, **kwargs):
        index = len(popen_calls)
        kwargs["stdout"].write(errors[index])
        kwargs["stdout"].flush()
        popen_calls.append((command, kwargs))
        return processes[index]

    monkeypatch.setattr(vllm_server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(vllm_server, "endpoint_is_ready", lambda *args, **kwargs: False)
    monkeypatch.setattr(vllm_server.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(vllm_server, "stop_process_group", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match=r"after 2 attempt\(s\)"):
        ManagedVllmServer.start(
            command=[sys.executable],
            project_root=tmp_path,
            log_path=tmp_path / "server.log",
            pid_path=tmp_path / "server.pid",
            base_url="http://127.0.0.1:8000/v1",
            expected_model="model",
            startup_timeout_seconds=5,
            shutdown_timeout_seconds=2,
        )

    assert len(popen_calls) == 2
