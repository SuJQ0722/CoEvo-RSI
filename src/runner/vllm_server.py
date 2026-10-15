"""Start, verify, and stop a local vLLM OpenAI-compatible server."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO
from urllib import error, request

_ADDRESS_IN_USE_MARKERS = ("EADDRINUSE", "address already in use")
_FATAL_STARTUP_MARKERS = (
    "DP Coordinator process failed",
    "Engine core initialization failed",
    "CUDA out of memory",
    "No available memory for the cache blocks",
)


def startup_log_has_address_in_use(log_path: Path, *, start_offset: int = 0) -> bool:
    """Return whether a failed vLLM startup hit a local port collision."""
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(start_offset, size - 256 * 1024))
            tail = handle.read().decode(errors="replace")
    except OSError:
        return False
    lowered = tail.lower()
    return any(marker.lower() in lowered for marker in _ADDRESS_IN_USE_MARKERS)


def startup_log_fatal_error(log_path: Path, *, start_offset: int = 0) -> str | None:
    """Return a known fatal startup marker even if vLLM children keep the process alive."""

    try:
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(start_offset, size - 256 * 1024))
            tail = handle.read().decode(errors="replace")
    except OSError:
        return None
    lowered = tail.lower()
    return next(
        (marker for marker in _FATAL_STARTUP_MARKERS if marker.lower() in lowered),
        None,
    )


def process_group_exists(pgid: int) -> bool:
    """Return whether a non-zombie process still belongs to ``pgid``."""
    proc_root = Path("/proc")
    if proc_root.is_dir():
        try:
            entries = tuple(proc_root.iterdir())
        except OSError:
            entries = ()
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="utf-8")
                fields = stat[stat.rfind(")") + 2 :].split()
                state, process_group = fields[0], int(fields[2])
            except (OSError, ValueError, IndexError):
                continue
            if process_group == pgid and state != "Z":
                return True
        return False

    # Non-procfs fallback. This may briefly count an unreaped zombie as alive.
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_process_group(pgid: int, *, timeout: float) -> None:
    """Stop a process group, including children orphaned by its leader."""
    if not process_group_exists(pgid):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + timeout
    while process_group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.25)
    if not process_group_exists(pgid):
        return

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    kill_deadline = time.monotonic() + max(2.0, min(timeout, 10.0))
    while process_group_exists(pgid) and time.monotonic() < kill_deadline:
        time.sleep(0.1)
    if process_group_exists(pgid):
        raise RuntimeError(f"vLLM process group {pgid} did not exit after SIGKILL")


def local_vllm_process_ids() -> set[int]:
    """Find live vLLM API, engine, and worker processes owned by this user."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return set()
    owner = os.getuid()
    matches: set[int] = set()
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        return matches
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            if pid == os.getpid() or entry.stat().st_uid != owner:
                continue
            stat = (entry / "stat").read_text(encoding="utf-8")
            state = stat[stat.rfind(")") + 2 :].split()[0]
            if state == "Z":
                continue
            comm = (entry / "comm").read_text(encoding="utf-8").strip()
            argv = [
                value.decode(errors="replace")
                for value in (entry / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except (OSError, ValueError, IndexError):
            continue
        command_is_vllm = any(Path(value).name == "vllm" for value in argv[:3])
        if comm.startswith("VLLM::") or command_is_vllm:
            matches.add(pid)
    return matches


def kill_local_vllm_processes(*, timeout: float = 10.0) -> list[int]:
    """Terminate all live vLLM processes owned by this user, then verify exit."""
    matched = local_vllm_process_ids()
    if not matched:
        return []
    for pid in matched:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = local_vllm_process_ids()
        matched.update(alive)
        if not alive:
            return sorted(matched)
        time.sleep(0.25)

    alive = local_vllm_process_ids()
    matched.update(alive)
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    kill_deadline = time.monotonic() + max(2.0, min(timeout, 10.0))
    while time.monotonic() < kill_deadline:
        alive = local_vllm_process_ids()
        if not alive:
            return sorted(matched)
        time.sleep(0.1)
    raise RuntimeError(
        "vLLM cleanup failed; processes still alive: "
        + ",".join(str(pid) for pid in sorted(local_vllm_process_ids()))
    )


def build_vllm_command(
    *,
    settings: dict[str, Any],
    model: str,
    served_model_name: str,
) -> list[str]:
    command = [
        str(settings.get("command") or "vllm"),
        "serve",
        model,
        "--host",
        str(settings.get("host") or "127.0.0.1"),
        "--port",
        str(int(settings.get("port", 8000))),
        "--served-model-name",
        served_model_name,
        "--max-model-len",
        str(int(settings.get("max_model_len", 8192))),
        "--gpu-memory-utilization",
        str(float(settings.get("gpu_memory_utilization", 0.85))),
        "--tensor-parallel-size",
        str(int(settings.get("tensor_parallel_size", 1))),
        "--data-parallel-size",
        str(int(settings.get("data_parallel_size", 1))),
    ]
    dtype = str(settings.get("dtype") or "auto")
    if dtype != "auto":
        command.extend(["--dtype", dtype])
    command.extend(str(value) for value in settings.get("extra_args", []))
    return command


def model_ids(
    base_url: str,
    timeout: float = 2.0,
    api_key: str | None = None,
) -> set[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    model_request = request.Request(
        f"{base_url.rstrip('/')}/models",
        headers=headers,
    )
    with request.urlopen(model_request, timeout=timeout) as response:
        payload = json.load(response)
    return {
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    }


def endpoint_is_ready(
    base_url: str,
    expected_model: str | None = None,
    *,
    api_key: str | None = None,
    timeout: float = 2.0,
) -> bool:
    try:
        models = model_ids(base_url, timeout=timeout, api_key=api_key)
    except (OSError, ValueError, error.URLError):
        return False
    return expected_model in models if expected_model else bool(models)


@dataclass
class ManagedVllmServer:
    process: subprocess.Popen
    log_handle: TextIO
    log_path: Path
    pid_path: Path
    base_url: str
    shutdown_timeout_seconds: float
    _detached: bool = False

    @classmethod
    def start(
        cls,
        *,
        command: list[str],
        project_root: Path,
        log_path: Path,
        pid_path: Path,
        base_url: str,
        expected_model: str,
        startup_timeout_seconds: float,
        shutdown_timeout_seconds: float,
        address_in_use_retries: int = 2,
    ) -> ManagedVllmServer:
        if address_in_use_retries < 0:
            raise ValueError("address_in_use_retries must be non-negative")
        executable = command[0]
        if not (Path(executable).is_file() or shutil.which(executable)):
            raise FileNotFoundError(f"vLLM executable not found: {executable}")
        if endpoint_is_ready(base_url):
            raise RuntimeError(f"refusing to replace an existing server at {base_url}")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.setdefault("HF_HUB_OFFLINE", "1")
        environment.setdefault("TRANSFORMERS_OFFLINE", "1")
        environment.setdefault("HF_DATASETS_OFFLINE", "1")
        total_attempts = address_in_use_retries + 1
        for attempt in range(1, total_attempts + 1):
            log_handle = log_path.open(
                "w" if attempt == 1 else "a", encoding="utf-8"
            )
            if attempt > 1:
                log_handle.write(
                    f"\n[CoEvoX] retrying vLLM startup after address-in-use "
                    f"failure (attempt {attempt}/{total_attempts})\n"
                )
                log_handle.flush()
            attempt_log_offset = log_path.stat().st_size
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
            server = cls(
                process=process,
                log_handle=log_handle,
                log_path=log_path,
                pid_path=pid_path,
                base_url=base_url,
                shutdown_timeout_seconds=shutdown_timeout_seconds,
            )
            deadline = time.monotonic() + startup_timeout_seconds
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    server.stop()
                    if (
                        attempt < total_attempts
                        and startup_log_has_address_in_use(
                            log_path, start_offset=attempt_log_offset
                        )
                    ):
                        time.sleep(1)
                        break
                    raise RuntimeError(
                        f"vLLM exited during startup after {attempt} attempt(s); "
                        f"inspect {log_path}"
                    )
                fatal_error = startup_log_fatal_error(
                    log_path, start_offset=attempt_log_offset
                )
                if fatal_error:
                    server.stop()
                    if (
                        attempt < total_attempts
                        and startup_log_has_address_in_use(
                            log_path, start_offset=attempt_log_offset
                        )
                    ):
                        time.sleep(1)
                        break
                    raise RuntimeError(
                        f"vLLM reported fatal startup error ({fatal_error}); "
                        f"inspect {log_path}"
                    )
                if endpoint_is_ready(base_url, expected_model):
                    return server
                time.sleep(2)
            else:
                server.stop()
                if (
                    attempt < total_attempts
                    and startup_log_has_address_in_use(
                        log_path, start_offset=attempt_log_offset
                    )
                ):
                    time.sleep(1)
                    continue
                raise TimeoutError(
                    f"vLLM did not become ready after {attempt} attempt(s); "
                    f"inspect {log_path}"
                )
        raise AssertionError("unreachable vLLM startup retry state")

    def stop(self) -> None:
        if self._detached:
            return
        try:
            # The API server can exit before vLLM's TP workers. Waiting only on
            # Popen would then leave GPU-owning orphan workers behind.
            stop_process_group(
                self.process.pid,
                timeout=self.shutdown_timeout_seconds,
            )
            if self.process.poll() is None:
                self.process.wait(timeout=2)
        finally:
            self.log_handle.close()
            self.pid_path.unlink(missing_ok=True)

    def detach(self) -> int:
        """Leave a verified final deployment running after the orchestrator exits."""
        self.log_handle.close()
        self._detached = True
        return self.process.pid

    def relabel(self, *, log_path: Path, pid_path: Path) -> None:
        """Promote a running validation server without restarting its process."""
        if self._detached:
            raise RuntimeError("cannot relabel a detached vLLM server")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.replace(log_path)
        self.pid_path.replace(pid_path)
        self.log_path = log_path
        self.pid_path = pid_path


def main() -> None:
    if len(sys.argv) not in (2, 4) or sys.argv[1] != "--kill-all":
        raise SystemExit(
            "usage: python -m src.runner.vllm_server --kill-all [--timeout SECONDS]"
        )
    timeout = 10.0
    if len(sys.argv) == 4:
        if sys.argv[2] != "--timeout":
            raise SystemExit("expected --timeout SECONDS")
        timeout = float(sys.argv[3])
    killed = kill_local_vllm_processes(timeout=timeout)
    print(
        "vLLM cleanup complete: "
        + ("killed=" + ",".join(map(str, killed)) if killed else "no_processes_found")
    )


if __name__ == "__main__":
    main()
