"""Direct CLI adapters for Claude Code and Codex."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.config import CANDIDATE_DIR, active_harness, harness_relative_path, harness_spec
from src.domains import get_domain_plugin
from src.meta_evolver.evolution import (
    ensure_parent_harness_registered,
    register_candidate,
    semantic_harness_hash,
    validate_candidate_bundle,
)
from src.meta_evolver.feedback import (
    build_failure_feedback,
    write_failure_feedback,
)
from src.meta_evolver.prompts import (
    DEFAULT_INSTRUCTION as DEFAULT_META_EVOLUTION_INSTRUCTION,
)
from src.meta_evolver.prompts import (
    build_prompt as build_meta_evolver_prompt,
)
from src.utils.persistence import atomic_json

_HARNESS_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class MetaEvolverResult:
    backend: str
    command: tuple[str, ...]
    run_dir: Path
    return_code: int
    parent_harness: str
    candidate_harness: str
    smoke_passed: bool | None


class MetaEvolverError(RuntimeError):
    """A bounded meta-evolver rejection with machine-readable feedback."""

    def __init__(self, message: str, run_dir: Path, feedback_path: Path):
        super().__init__(message)
        self.run_dir = run_dir
        self.feedback_path = feedback_path


def build_cli_command(
    backend: str,
    settings: dict[str, Any],
    project_root: Path,
    final_message_path: Path,
    output_schema_path: Path | None = None,
) -> list[str]:
    command = str(settings.get("command") or backend)
    extra_args = [str(value) for value in settings.get("extra_args", [])]
    if backend == "codex":
        sandbox = str(settings.get("sandbox", "workspace-write"))
        if sandbox != "workspace-write":
            raise ValueError("Codex meta-evolution requires sandbox=workspace-write")
        unsafe_prefixes = (
            "--add-dir",
            "--cd",
            "-C",
            "--config",
            "-c",
            "--dangerously-bypass",
            "--profile",
            "-p",
            "--sandbox",
            "-s",
        )
        unsafe = [
            value
            for value in extra_args
            if any(value == prefix or value.startswith(prefix + "=") for prefix in unsafe_prefixes)
        ]
        if unsafe:
            raise ValueError(f"unsafe Codex meta-evolver extra_args: {unsafe}")
        result = [
            command,
            "exec",
            "--sandbox",
            sandbox,
            "-c",
            'approval_policy="never"',
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ignore-rules",
            "--strict-config",
        ]
        if settings.get("multi_agent", True):
            result[2:2] = ["--enable", "multi_agent"]
        reasoning_effort = str(settings.get("reasoning_effort") or "medium")
        if reasoning_effort not in {"minimal", "low", "medium", "high", "xhigh"}:
            raise ValueError(
                "meta_evolver.codex.reasoning_effort must be minimal, low, medium, high, or xhigh"
            )
        result.extend(["-c", f"model_reasoning_effort={json.dumps(reasoning_effort)}"])
        base_url = str(settings.get("base_url") or "").strip()
        api_key_env = str(settings.get("api_key_env") or "").strip()
        if base_url:
            if not api_key_env:
                raise ValueError("custom Codex base_url requires api_key_env")
            provider = "coevox_custom"
            result.extend(
                [
                    "-c",
                    f"model_provider={json.dumps(provider)}",
                    "-c",
                    f"model_providers.{provider}.name={json.dumps('CoEvoX custom')}",
                    "-c",
                    f"model_providers.{provider}.base_url={json.dumps(base_url)}",
                    "-c",
                    f"model_providers.{provider}.env_key={json.dumps(api_key_env)}",
                    "-c",
                    f"model_providers.{provider}.wire_api={json.dumps('responses')}",
                ]
            )
        if settings.get("ephemeral", True):
            result.append("--ephemeral")
        model = str(settings.get("model") or "").strip()
        if model:
            result.extend(["--model", model])
        if output_schema_path is not None:
            result.extend(["--output-schema", str(output_schema_path)])
        result.extend(
            [
                "--json",
                "-C",
                str(project_root),
                "--output-last-message",
                str(final_message_path),
                *extra_args,
                "-",
            ]
        )
        return result
    if backend == "claude":
        return [
            command,
            "-p",
            "--permission-mode",
            str(settings.get("permission_mode", "acceptEdits")),
            *extra_args,
        ]
    raise ValueError(f"unsupported meta-evolver backend: {backend}")


def _subprocess_environment(
    settings: dict[str, Any],
    codex_home: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    api_key = str(settings.get("api_key") or "")
    api_key_env = str(settings.get("api_key_env") or "")
    if api_key and api_key_env:
        environment[api_key_env] = api_key
    if codex_home is not None:
        codex_home.mkdir(parents=True, exist_ok=True)
        environment["CODEX_HOME"] = str(codex_home)
    if project_root is not None:
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            str(project_root) if not existing else str(project_root) + os.pathsep + existing
        )
    return environment


def _require_command(command: str) -> None:
    if Path(command).is_file() or shutil.which(command):
        return
    raise FileNotFoundError(
        f"meta-evolver executable not found: {command}. Install it or update configs/base.yaml"
    )


def _copy_bounded_evidence(source: Path, target: Path, limit: int | None) -> int:
    copied = 0
    with (
        source.open(encoding="utf-8") as input_handle,
        target.open("w", encoding="utf-8") as output_handle,
    ):
        for line in input_handle:
            if not line.strip():
                continue
            json.loads(line)
            output_handle.write(line if line.endswith("\n") else line + "\n")
            copied += 1
            if limit is not None and copied >= limit:
                break
    if not copied:
        raise ValueError(f"rollout evidence is empty: {source}")
    return copied


def _prepare_trajectory_directory(source: Path, target: Path) -> Path:
    """Stage one rollout or all top-level JSONL rollouts as read-only evidence."""
    target.mkdir(parents=True, exist_ok=False)
    sources = (
        [source]
        if source.is_file()
        else sorted(path for path in source.iterdir() if path.is_file() and path.suffix == ".jsonl")
    )
    if not sources:
        raise ValueError(f"rollout evidence directory is empty: {source}")
    for path in sources:
        shutil.copy2(path, target / path.name)
    return target / sources[0].name if source.is_file() else target


def _write_meta_failure_feedback(
    run_dir: Path,
    *,
    status: str,
    reason: str,
    causes: list[str],
    evidence: dict[str, Any],
    retryable: bool,
) -> Path:
    path = run_dir / "failure_feedback.json"
    return write_failure_feedback(
        path,
        build_failure_feedback(
            status=status,
            reason=reason,
            causes=causes,
            evidence=evidence,
            retryable=retryable,
        ),
    )


def _parent_harness_path(harness_root: Path, name: str) -> Path:
    if not _HARNESS_NAME.fullmatch(name):
        raise ValueError(f"invalid parent harness name: {name!r}")
    relative = harness_relative_path(name)
    path = (harness_root / relative).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"parent harness directory not found: {path}")
    return path


def _copy_candidate_source(source: Path, target: Path) -> None:
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def _prepare_reference_files(
    project_root: Path,
    reference_root: Path,
    parent_harness: Path,
) -> None:
    reference_root.mkdir(parents=True)
    sources = {
        "harness_contract.py": parent_harness / "__init__.py",
    }
    prompt_reference = parent_harness / "prompts.py"
    if prompt_reference.is_file():
        sources["prompts_reference.py"] = prompt_reference
    for name, source in sources.items():
        shutil.copy2(source, reference_root / name)


def _make_read_only(path: Path) -> None:
    """Enforce the prompt's read-only evidence/reference boundary at the OS layer."""
    if not path.exists():
        return
    if path.is_file():
        path.chmod(0o444)
        return
    for child in sorted(path.rglob("*"), reverse=True):
        child.chmod(0o555 if child.is_dir() else 0o444)
    path.chmod(0o555)


def _latest_failure_feedback(
    registry_path: Path, project_root: Path, parent_name: str
) -> dict[str, Any] | None:
    if not registry_path.is_file():
        return None
    tree = json.loads(registry_path.read_text(encoding="utf-8"))
    candidates: list[tuple[str, str]] = []
    for node in tree.get("nodes", {}).values():
        if not isinstance(node, dict) or node.get("parent") != parent_name:
            continue
        for selection in node.get("selections", []):
            if selection.get("decision") != "promote_child":
                candidates.append(
                    (str(selection.get("evaluated_at", "")), str(selection.get("report", "")))
                )
    for _, report_value in sorted(candidates, reverse=True):
        report_path = Path(report_value)
        if not report_path.is_absolute():
            report_path = project_root / report_path
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            feedback = report.get("failure_feedback")
            if isinstance(feedback, dict):
                return feedback
    return None


def _file_snapshot(root: Path) -> dict[str, str]:
    import hashlib

    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            relative = path.relative_to(root).as_posix()
            snapshot[relative] = f"SYMLINK:{os.readlink(path)}"
        elif path.is_file():
            relative = path.relative_to(root).as_posix()
            snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def _workspace_permission_violations(before: dict[str, str], after: dict[str, str]) -> list[str]:
    violations: list[str] = []
    for relative in sorted(set(before) | set(after)):
        if relative.startswith("candidate/") or relative == "final_message.md":
            continue
        if before.get(relative) != after.get(relative):
            violations.append(relative)
    return violations


def _analysis_report_violations(path: Path) -> list[str]:
    if not path.is_file():
        return ["meta-evolver did not produce final_message.md"]
    report = path.read_text(encoding="utf-8").strip()
    violations: list[str] = []
    if len(report) < 600:
        violations.append("evolution report is incomplete (minimum 600 characters)")
    if "candidate/" not in report:
        violations.append("evolution report does not identify changed candidate files")
    return violations


def run_meta_evolver(
    config: dict[str, Any],
    project_root: Path,
    trace_path: Path,
    backend: str | None = None,
    instruction: str | None = None,
    candidate_name: str | None = None,
    prior_analysis_path: Path | None = None,
    prior_feedback_path: Path | None = None,
    seed_candidate_path: Path | None = None,
    dry_run: bool = False,
    batch_context_path: Path | None = None,
    brief_path: Path | None = None,
    run_dir_override: Path | None = None,
    parent_source_path: Path | None = None,
    publish_registry: bool = True,
) -> MetaEvolverResult:
    meta = config.get("meta_evolver")
    if not isinstance(meta, dict):
        raise TypeError("configs/base.yaml is missing meta_evolver")
    selected = backend or str(meta.get("backend", "codex"))
    settings = meta.get(selected)
    if not isinstance(settings, dict):
        raise TypeError(f"missing meta_evolver.{selected} configuration")
    if selected == "claude" and not settings.get("external_sandbox"):
        raise RuntimeError(
            "Claude meta-evolution is disabled without an externally enforced "
            "candidate-only write sandbox"
        )

    lineage_editor = batch_context_path is not None
    if lineage_editor:
        if (
            brief_path is None
            or seed_candidate_path is not None
            or prior_analysis_path
            or prior_feedback_path
        ):
            raise ValueError(
                "LINEAGE Editor requires its own brief and frozen context, without seed/implicit feedback"
            )
        settings = dict(settings, multi_agent=False, ephemeral=True)
    project_root = project_root.resolve()
    selected_harness = harness_spec(config, project_root=project_root)
    domain = get_domain_plugin(config)
    trace_path = trace_path.resolve()
    if not dry_run and not trace_path.is_file() and not trace_path.is_dir():
        raise FileNotFoundError(f"rollout file or directory not found: {trace_path}")
    output_root = project_root / str(meta.get("output_dir", "output/meta_evolver"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parent_name = active_harness(config)
    candidate_name = candidate_name or f"candidate_{timestamp}_{uuid4().hex[:8]}"
    if not _HARNESS_NAME.fullmatch(candidate_name):
        raise ValueError(f"invalid candidate harness name: {candidate_name!r}")
    run_dir = run_dir_override or output_root / f"{timestamp}_{selected}_{uuid4().hex[:8]}"
    workspace_root = run_dir / "workspace"
    candidate_workspace = workspace_root / "candidate"
    evidence_mode = str(meta.get("evidence_mode", "all"))
    if evidence_mode not in {"all", "trajectory_directory"}:
        raise ValueError("meta_evolver.evidence_mode must be all or trajectory_directory")
    # Every domain normalizes its native val artifacts into the same single
    # evidence envelope before the MetaAgent sees them.
    evidence_path = workspace_root / "evidence.jsonl"
    supplied_prior_feedback_path = prior_feedback_path
    copied_prior_analysis_path = workspace_root / "prior_analysis.md"
    final_message_path = workspace_root / "final_message.md"
    command = build_cli_command(selected, settings, workspace_root, final_message_path)
    registry_path = project_root / str(meta.get("registry_path", "registries/harness_tree.json"))
    prior_feedback = None
    if supplied_prior_feedback_path is None and not lineage_editor:
        prior_feedback = _latest_failure_feedback(registry_path, project_root, parent_name)
    max_trace_items_value = meta.get("max_trace_items", 50)
    max_trace_items = None if max_trace_items_value is None else int(max_trace_items_value)
    if max_trace_items is not None and max_trace_items < 1:
        raise ValueError("meta_evolver.max_trace_items must be null or positive")
    prompt = build_meta_evolver_prompt(
        workspace_root=workspace_root,
        evidence_path=evidence_path,
        candidate_name=candidate_name,
        parent_name=parent_name,
        instruction=instruction or DEFAULT_META_EVOLUTION_INSTRUCTION,
        max_trace_items=max_trace_items,
        evidence_mode=evidence_mode,
        prior_feedback_path=(
            workspace_root / "prior_failure_feedback.json"
            if prior_feedback is not None or supplied_prior_feedback_path is not None
            else None
        ),
        prior_analysis_path=(
            copied_prior_analysis_path if prior_analysis_path is not None else None
        ),
        domain_user_prompt=domain.meta_user_prompt(config),
        allow_research_subagent=bool(settings.get("multi_agent", True)),
        resume_seeded=(seed_candidate_path is not None and prior_analysis_path is not None),
    )

    if lineage_editor:
        from src.meta_evolver.context import editor_prompt

        prompt = editor_prompt(candidate_name, parent_name)
    if dry_run:
        return MetaEvolverResult(
            selected,
            tuple(command),
            run_dir,
            0,
            parent_name,
            candidate_name,
            None,
        )

    _require_command(command[0])
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(exist_ok=True)
    if lineage_editor:
        for source, name in (
            (batch_context_path, "batch_context.json"),
            (brief_path, "brief.json"),
        ):
            target = workspace_root / name
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError("frozen Editor input changed on resume")
            if not target.exists():
                shutil.copy2(source, target)
            _make_read_only(target)
    else:
        normalized_evidence = run_dir / "normalized_evidence.jsonl"
        domain.build_evidence(trace_path, normalized_evidence)
        _copy_bounded_evidence(normalized_evidence, evidence_path, max_trace_items)
        normalized_evidence.unlink()
    workspace_feedback_path = workspace_root / "prior_failure_feedback.json"
    if prior_feedback is not None:
        atomic_json(workspace_feedback_path, prior_feedback)
    elif supplied_prior_feedback_path is not None:
        supplied_prior_feedback_path = supplied_prior_feedback_path.resolve()
        if not supplied_prior_feedback_path.is_file():
            raise FileNotFoundError(f"prior feedback not found: {supplied_prior_feedback_path}")
        shutil.copy2(supplied_prior_feedback_path, workspace_feedback_path)
    if prior_analysis_path is not None:
        prior_analysis_path = prior_analysis_path.resolve()
        if not prior_analysis_path.is_file():
            raise FileNotFoundError(f"prior analysis not found: {prior_analysis_path}")
        shutil.copy2(prior_analysis_path, copied_prior_analysis_path)
    parent_harness_path = parent_source_path or _parent_harness_path(
        selected_harness.root,
        parent_name,
    )
    candidate_source = parent_harness_path
    if seed_candidate_path is not None:
        candidate_source = seed_candidate_path.resolve()
        if not candidate_source.is_dir():
            raise FileNotFoundError(f"seed candidate directory not found: {candidate_source}")
        seed_violations = validate_candidate_bundle(candidate_source)
        if seed_violations:
            raise ValueError(
                "seed candidate failed bundle validation: " + "; ".join(seed_violations)
            )
    if not (workspace_root / "reference").exists():
        _prepare_reference_files(project_root, workspace_root / "reference", parent_harness_path)
    if not candidate_workspace.exists():
        _copy_candidate_source(candidate_source, candidate_workspace)
    for protected_path in (
        evidence_path,
        workspace_root / "reference",
        workspace_feedback_path,
        copied_prior_analysis_path,
    ):
        _make_read_only(protected_path)
    parent_semantic_hash = semantic_harness_hash(parent_harness_path)
    before = _file_snapshot(workspace_root)
    if lineage_editor:
        original_boundary = run_dir / "input_boundary.json"
        if original_boundary.exists():
            before = json.loads(original_boundary.read_text())
        else:
            atomic_json(original_boundary, before)
    try:
        receipt = run_dir / "process_result.json"
        if lineage_editor and receipt.exists():
            saved = json.loads(receipt.read_text())
            completed = subprocess.CompletedProcess(
                command,
                saved["return_code"],
                (run_dir / "stdout.log").read_text(),
                (run_dir / "stderr.log").read_text(),
            )
        else:
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                cwd=workspace_root,
                env=_subprocess_environment(
                    settings,
                    run_dir / "codex_home" if selected == "codex" else None,
                    project_root,
                ),
                timeout=int(meta.get("timeout_seconds", 7200)),
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        feedback_path = _write_meta_failure_feedback(
            run_dir,
            status="meta_evolver_timeout",
            reason=f"{selected} exceeded the configured timeout",
            causes=["meta_evolver_timeout"],
            evidence={"timeout_seconds": int(meta.get("timeout_seconds", 7200))},
            retryable=False,
        )
        raise MetaEvolverError(
            f"{selected} meta-evolver timed out; see {run_dir}",
            run_dir,
            feedback_path,
        ) from exc
    (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if not final_message_path.is_file():
        final_message_path.write_text(completed.stdout, encoding="utf-8")
    if lineage_editor:
        atomic_json(run_dir / "process_result.json", {"return_code": completed.returncode})
    permission_violations = _workspace_permission_violations(before, _file_snapshot(workspace_root))
    bundle_violations = validate_candidate_bundle(candidate_workspace)
    # MetaAgents are instructed to author the durable report inside the
    # candidate bundle. Their concise final response often only links to that
    # report, so validate the report itself when present instead of rejecting
    # a valid child based on its chat summary.
    candidate_report_path = candidate_workspace / "evolution_report.md"
    report_source = candidate_report_path if candidate_report_path.is_file() else final_message_path
    report_violations = _analysis_report_violations(report_source)
    child_semantic_hash = semantic_harness_hash(candidate_workspace)
    if child_semantic_hash == parent_semantic_hash:
        bundle_violations.append("candidate has no semantic Python changes relative to its parent")
    manifest = {
        "backend": selected,
        "command": command,
        "trace_path": str(trace_path),
        "evidence_mode": evidence_mode,
        "evidence_path": str(evidence_path),
        "parent_harness": parent_name,
        "seed_candidate": str(candidate_source) if seed_candidate_path else None,
        "candidate_harness": candidate_name,
        "return_code": completed.returncode,
        "permission_violations": permission_violations,
        "bundle_violations": bundle_violations,
        "report_violations": report_violations,
        "parent_semantic_hash": parent_semantic_hash,
        "child_semantic_hash": child_semantic_hash,
    }
    if completed.returncode:
        atomic_json(run_dir / "manifest.json", manifest)
        feedback_path = _write_meta_failure_feedback(
            run_dir,
            status="meta_evolver_failed",
            reason=f"{selected} exited with code {completed.returncode}",
            causes=["meta_evolver_nonzero_exit"],
            evidence={"return_code": completed.returncode},
            retryable=True,
        )
        raise MetaEvolverError(
            f"{selected} meta-evolver failed with exit code {completed.returncode}; see {run_dir}",
            run_dir,
            feedback_path,
        )
    if permission_violations or bundle_violations or report_violations:
        atomic_json(run_dir / "manifest.json", manifest)
        feedback_path = _write_meta_failure_feedback(
            run_dir,
            status="candidate_write_boundary_rejected",
            reason="candidate rejected by the write boundary",
            causes=[
                *(["workspace_permission_violation"] if permission_violations else []),
                *(["candidate_bundle_violation"] if bundle_violations else []),
                *(["incomplete_evolution_report"] if report_violations else []),
            ],
            evidence={
                "permission_violations": permission_violations,
                "bundle_violations": bundle_violations,
                "report_violations": report_violations,
            },
            retryable=True,
        )
        raise MetaEvolverError(
            f"candidate rejected by the write boundary; see {run_dir / 'manifest.json'}",
            run_dir,
            feedback_path,
        )

    if report_source != candidate_report_path:
        shutil.copy2(report_source, candidate_report_path)
    destination = selected_harness.root / CANDIDATE_DIR / candidate_name
    if destination.exists():
        if not lineage_editor or semantic_harness_hash(destination) != child_semantic_hash:
            raise FileExistsError(f"candidate destination already exists: {destination}")
    else:
        _copy_candidate_source(candidate_workspace, destination)
    smoke = domain.smoke_test(
        candidate_name,
        package=selected_harness.package,
        root=selected_harness.root,
    )
    manifest["smoke"] = smoke.to_dict()
    manifest["evolution_report"] = str(destination / "evolution_report.md")
    atomic_json(run_dir / "manifest.json", manifest)
    try:
        relative_destination = destination.relative_to(project_root)
    except ValueError:
        relative_destination = destination
    if publish_registry:
        ensure_parent_harness_registered(
            registry_path,
            parent_name,
            parent_harness_path,
        )
        register_candidate(
            registry_path,
            candidate_name,
            parent_name,
            relative_destination,
            smoke,
        )
    if not smoke.passed:
        feedback_path = _write_meta_failure_feedback(
            run_dir,
            status="candidate_smoke_rejected",
            reason=f"candidate {candidate_name} failed smoke tests",
            causes=["child_failed_smoke_contract"],
            evidence={"smoke": smoke.to_dict()},
            retryable=True,
        )
        raise MetaEvolverError(
            f"candidate {candidate_name} failed smoke tests and remains inactive; "
            f"see {run_dir / 'manifest.json'}",
            run_dir,
            feedback_path,
        )
    return MetaEvolverResult(
        selected,
        tuple(command),
        run_dir,
        completed.returncode,
        parent_name,
        candidate_name,
        True,
    )
