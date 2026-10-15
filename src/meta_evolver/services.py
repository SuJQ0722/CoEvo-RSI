"""Explicit process, evaluation and training-preparation services for LINEAGE.

The controller depends on RuntimeServices, never on a CLI module. Resource
ownership/cleanup policy is implemented here and matches the existing launcher.
"""

from __future__ import annotations

import csv
import shlex
import subprocess
from pathlib import Path
from typing import Protocol

from src.domains.io import read_records
from src.domains.round_adapter import EvaluationArtifact
from src.runner.vllm_server import ManagedVllmServer, kill_local_vllm_processes
from src.training.preparation import prepare_training_dataset
from src.utils.logging import get_logger

LOGGER = get_logger("pipeline.round")


class RuntimeServices(Protocol):
    training_python: str

    def project_path(self, value: str | Path) -> Path: ...
    def run(
        self, command: list[str], *, module: str, dry_run: bool, check: bool = True
    ) -> subprocess.CompletedProcess | None: ...
    def start_vllm(self, command: list[str], **kwargs) -> ManagedVllmServer | None: ...
    def stop_vllm(self, server: ManagedVllmServer | None, *, label: str, dry_run: bool) -> None: ...
    def evaluate_with_adapter(self, adapter, **kwargs): ...
    def prepare_training_dataset(self, **kwargs) -> tuple[int, Path]: ...


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def command_logger_name(command: list[str]) -> str:
    """Return the Python module (or service wrapper) executed by a command."""
    try:
        module = command[command.index("-m") + 1]
    except (ValueError, IndexError):
        executable = Path(command[0]).name if command else "unknown"
        return "runner.vllm_server" if executable == "vllm" else executable
    return module.removeprefix("src.")


def expected_sample_ids(path: Path, limit: int | None = None) -> list[str]:
    """Return the ordered sample IDs consumed by runner.infer."""
    sample_ids: list[str] = []
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for index, record in enumerate(csv.DictReader(stream), start=1):
                if limit is not None and index > limit:
                    break
                sample_ids.append(str(record.get("sample_id") or f"{path.stem}:{index:06d}"))
        return sample_ids
    for index, record in enumerate(read_records(path)):
        if limit is not None and index >= limit:
            break
        sample_id = str(
            record.get("sample_id") or record.get("source_id") or record.get("id") or ""
        )
        if not sample_id:
            raise ValueError(f"dataset record has no sample ID: {path}:{index + 1}")
        sample_ids.append(sample_id)
    return sample_ids


class LocalRuntimeServices:
    def __init__(self, project_root: Path, *, training_python: str):
        self.project_root = project_root.resolve()
        self.training_python = training_python

    def prepare_training_dataset(self, **kwargs) -> tuple[int, Path]:
        return prepare_training_dataset(**kwargs)

    def project_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def portable_path(self, path: Path | None) -> str | None:
        """Use repository-relative paths in persisted metadata whenever possible."""
        if path is None:
            return None
        try:
            return path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return path.as_posix()

    def run(
        self, command: list[str], *, module: str, dry_run: bool, check: bool = True
    ) -> subprocess.CompletedProcess | None:
        command_logger = get_logger(command_logger_name(command))
        command_logger.info("start stage=%s", module)
        command_logger.info("stage=%s command=%s", module, command_text(command))
        if dry_run:
            return None
        completed = subprocess.run(command, cwd=self.project_root, check=False)
        if check and completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, command)
        command_logger.info("complete stage=%s return_code=%d", module, completed.returncode)
        return completed

    def start_vllm(
        self,
        command: list[str],
        *,
        label: str,
        round_dir: Path,
        base_url: str,
        expected_model: str,
        settings: dict,
        dry_run: bool,
    ) -> ManagedVllmServer | None:
        server_logger = get_logger("runner.vllm_server")
        server_logger.info("start stage=%s", label)
        server_logger.info("stage=%s command=%s", label, command_text(command))
        if dry_run:
            return None
        killed = kill_local_vllm_processes(
            timeout=float(settings.get("cleanup_timeout_seconds", 10))
        )
        if killed:
            server_logger.warning(
                "preflight cleanup stage=%s killed_pids=%s",
                label,
                ",".join(map(str, killed)),
            )
        else:
            server_logger.info("preflight cleanup stage=%s no_processes_found", label)
        return ManagedVllmServer.start(
            command=command,
            project_root=self.project_root,
            log_path=round_dir / f"{label}_server.log",
            pid_path=round_dir / f"{label}_server.pid",
            base_url=base_url,
            expected_model=expected_model,
            startup_timeout_seconds=float(settings.get("startup_timeout_seconds", 900)),
            shutdown_timeout_seconds=float(settings.get("shutdown_timeout_seconds", 60)),
        )

    def stop_vllm(self, server: ManagedVllmServer | None, *, label: str, dry_run: bool) -> None:
        get_logger("runner.vllm_server").info("stop stage=%s", label)
        if not dry_run and server is not None:
            server.stop()

    def evaluate_with_adapter(
        self,
        adapter,
        *,
        python: str,
        config: Path,
        data: Path,
        split: str,
        model: str,
        base_url: str,
        harness: str,
        max_turns: int,
        concurrency: int,
        artifact: EvaluationArtifact,
        limit: int | None,
        resume: bool,
        stage: str,
        dry_run: bool,
    ):
        if data.is_file():
            expected = len(expected_sample_ids(data, limit))
        elif dry_run:
            expected = int(limit or 0)
        else:
            expected = len(expected_sample_ids(data, limit))
        if resume and adapter.artifact_complete(artifact, expected):
            LOGGER.info(
                "resume skip stage=%s records=%d output=%s",
                stage,
                expected,
                self.portable_path(artifact.root),
            )
        else:
            for command, suffix in adapter.evaluation_commands(
                python=python,
                config=config,
                data=data,
                split=split,
                model=model,
                base_url=base_url,
                harness=harness,
                max_turns=max_turns,
                concurrency=concurrency,
                artifact=artifact,
                limit=limit,
                resume=resume,
            ):
                self.run(command, module=f"{stage}.{suffix}", dry_run=dry_run)
        return None if dry_run else adapter.load_result(artifact)
