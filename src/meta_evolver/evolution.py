"""Safety, lineage, smoke-test, and Pareto primitives for harness evolution."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.domains.base import SmokeCheck as SmokeCheck
from src.domains.base import SmokeReport
from src.utils.persistence import atomic_json
from src.utils.python_source import normalized_python

ALLOWED_CANDIDATE_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".txt", ".sh"}
ALLOWED_CANDIDATE_FILENAMES = {"Dockerfile"}
_SIBLING_IMPORT = re.compile(r"\bharnesses\.\w+\.(?:candidates|generated)\b")


def _module_path(root: Path, parts: tuple[str, ...]) -> Path | None:
    module_file = root.joinpath(*parts).with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_file = root.joinpath(*parts, "__init__.py")
    return package_file if package_file.is_file() else None


def _reachable_python_files(harness_path: Path) -> list[Path]:
    """Resolve local relative imports from the harness package entry point."""
    entrypoint = harness_path / "__init__.py"
    if not entrypoint.is_file():
        return []
    pending = [entrypoint]
    reachable: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in reachable:
            continue
        reachable.add(path)
        relative = path.relative_to(harness_path)
        module_parts = relative.with_suffix("").parts
        package_parts = relative.parent.parts if path.name == "__init__.py" else module_parts[:-1]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level < 1:
                continue
            ascend = node.level - 1
            if ascend > len(package_parts):
                continue
            base = package_parts[: len(package_parts) - ascend]
            target = base + tuple((node.module or "").split("."))
            target = tuple(part for part in target if part)
            candidates = [target]
            candidates.extend(target + (alias.name,) for alias in node.names if alias.name != "*")
            for parts in candidates:
                imported = _module_path(harness_path, parts)
                if imported is not None and imported not in reachable:
                    pending.append(imported)
                for depth in range(1, len(parts)):
                    package_init = harness_path.joinpath(*parts[:depth], "__init__.py")
                    if package_init.is_file() and package_init not in reachable:
                        pending.append(package_init)
    return sorted(reachable)


def semantic_harness_hash(harness_path: Path) -> str:
    """Hash statically reachable Python from __init__, using AST normalization v1.

    Candidate de-duplication only: excludes unreferenced modules and all assets.
    Keep the path/NUL/AST/NUL stream stable for historical semantic identities.
    """
    digest = hashlib.sha256()
    for path in _reachable_python_files(harness_path):
        normalized = normalized_python(path.read_text(encoding="utf-8"), filename=str(path))
        digest.update(path.relative_to(harness_path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_candidate_bundle(candidate_path: Path) -> list[str]:
    """Return blocking violations without importing or executing candidate code."""
    candidate_path = candidate_path.resolve()
    violations: list[str] = []
    if not (candidate_path / "__init__.py").is_file():
        violations.append("candidate must contain __init__.py")
    for path in sorted(candidate_path.rglob("*")):
        relative = path.relative_to(candidate_path).as_posix()
        if path.is_symlink():
            violations.append(f"symlink is not allowed: {relative}")
            continue
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if (
            path.suffix not in ALLOWED_CANDIDATE_SUFFIXES
            and path.name not in ALLOWED_CANDIDATE_FILENAMES
        ):
            violations.append(f"unsupported candidate file: {relative}")
            continue
        if path.stat().st_size > 1_000_000:
            violations.append(f"candidate file exceeds 1 MB: {relative}")
        if path.suffix != ".py":
            continue
        source = path.read_text(encoding="utf-8")
        if _SIBLING_IMPORT.search(source):
            violations.append(f"sibling generated-harness import is not allowed: {relative}")
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            violations.append(f"Python syntax error in {relative}: {exc}")
            continue
    return violations


def load_harness_tree(path: Path) -> dict[str, Any]:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), dict):
            raise ValueError(f"invalid harness tree registry: {path}")
        # Registries created before the harness package migration use
        # ``system_zero`` as their root.  The current runtime calls the same
        # seed harness ``initial_harness``.  Keep the historical nodes intact,
        # but make the current seed available as a root so a resumed run can
        # register new children against it.
        if "initial_harness" not in payload["nodes"]:
            payload["nodes"]["initial_harness"] = {
                "parent": None,
                "generation": 0,
                "status": (
                    "active" if payload.get("active") in (None, "initial_harness") else "available"
                ),
                "path": None,
                "smoke": None,
                "selections": [],
            }
        return payload
    return {
        "schema_version": 1,
        "active": "initial_harness",
        "nodes": {
            "initial_harness": {
                "parent": None,
                "generation": 0,
                "status": "active",
                "path": None,
                "smoke": None,
                "selections": [],
            }
        },
    }


def write_harness_tree(path: Path, tree: dict[str, Any]) -> None:
    atomic_json(path, tree, sort_keys=True)


@contextmanager
def _harness_tree_lock(registry_path: Path):
    """Serialize registry read-modify-write cycles across MetaAgent processes."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def register_candidate(
    registry_path: Path,
    candidate_name: str,
    parent_name: str,
    candidate_path: Path,
    smoke: SmokeReport,
) -> None:
    with _harness_tree_lock(registry_path):
        tree = load_harness_tree(registry_path)
        nodes = tree["nodes"]
        if candidate_name in nodes:
            raise ValueError(f"harness already registered: {candidate_name}")
        if parent_name not in nodes:
            raise ValueError(f"parent harness is not registered: {parent_name}")
        nodes[candidate_name] = {
            "parent": parent_name,
            "generation": int(nodes[parent_name]["generation"]) + 1,
            "status": "smoke_passed" if smoke.passed else "smoke_failed",
            "path": candidate_path.as_posix(),
            "created_at": _utc_now(),
            "smoke": smoke.to_dict(),
            "selections": [],
        }
        write_harness_tree(registry_path, tree)


def ensure_parent_harness_registered(
    registry_path: Path,
    harness_name: str,
    harness_path: Path,
) -> None:
    """Ensure a domain registry points at its real active parent bundle.

    Empty registry nodes have no assumed path. Domain runners bind the actual
    configured parent before registering candidates.
    """

    with _harness_tree_lock(registry_path):
        tree = load_harness_tree(registry_path)
        project_root = (
            registry_path.parent.parent
            if registry_path.parent.name == "registries"
            else registry_path.parent
        )
        try:
            portable = harness_path.resolve().relative_to(project_root.resolve())
        except ValueError:
            portable = harness_path.resolve()
        node = tree["nodes"].get(harness_name)
        if not isinstance(node, dict):
            node = {
                "parent": None,
                "generation": 0,
                "status": "active",
                "selections": [],
                "smoke": None,
            }
            tree["nodes"][harness_name] = node
        node["path"] = portable.as_posix()
        node.setdefault("selections", [])
        tree.setdefault("active", harness_name)
        write_harness_tree(registry_path, tree)


def record_selection(
    registry_path: Path,
    parent_name: str,
    child_name: str,
    decision: str,
    report_path: Path,
) -> None:
    tree = load_harness_tree(registry_path)
    nodes = tree["nodes"]
    if parent_name not in nodes or child_name not in nodes:
        raise ValueError("selection references an unregistered harness")
    project_root = (
        registry_path.parent.parent
        if registry_path.parent.name == "registries"
        else registry_path.parent
    )
    try:
        portable_report_path = report_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        portable_report_path = report_path
    entry = {
        "evaluated_at": _utc_now(),
        "parent": parent_name,
        "child": child_name,
        "decision": decision,
        "report": portable_report_path.as_posix(),
    }
    nodes[child_name].setdefault("selections", []).append(entry)
    if decision in {"promote_child", "promote_pre_sft_harness"}:
        nodes[parent_name]["status"] = "archived"
        nodes[child_name]["status"] = "active"
        tree["active"] = child_name
    elif decision == "retain_parent":
        nodes[parent_name]["status"] = "active"
        nodes[child_name]["status"] = "rejected"
        tree["active"] = parent_name
    else:
        nodes[parent_name]["status"] = "active"
        nodes[child_name]["status"] = "pareto_candidate"
        tree["active"] = parent_name
    write_harness_tree(registry_path, tree)


def publish_lineage_harnesses(
    registry_path, *, parent, parent_path, proposals, winner, report_path
):
    """Idempotently publish a completed round in one legacy-registry transaction."""
    with _harness_tree_lock(registry_path):
        tree = load_harness_tree(registry_path)
        nodes = tree["nodes"]
        nodes.setdefault(parent, {"generation": 0, "parent": None, "selections": []})
        nodes[parent].update(
            path=str(parent_path), status="active" if winner == parent else "archived"
        )
        for proposal in proposals:
            name = proposal["candidate"]
            node = nodes.setdefault(
                name,
                {
                    "parent": parent,
                    "generation": int(nodes[parent].get("generation", 0)) + 1,
                    "selections": [],
                },
            )
            node.update(
                status="active" if name == winner else "rejected",
                path=proposal.get("source_path"),
                source_hash=proposal.get("harness_hash"),
                smoke=proposal.get("smoke"),
                brief_id=proposal.get("brief_id"),
                batch_context_id=proposal.get("batch_context_id"),
            )
            entry = {
                "parent": parent,
                "child": name,
                "decision": "promote_child" if name == winner else "retain_parent",
                "report": str(report_path),
            }
            if entry not in node["selections"]:
                node["selections"].append(entry)
        tree["active"] = winner
        write_harness_tree(registry_path, tree)
