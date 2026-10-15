"""Run the final model--harness pair from one evolution on its full test split."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import (
    apply_harness_overrides,
    harness_spec,
    load_config,
    resolve_output_root,
    write_resolved_config,
)
from src.domains.round_adapter import EvaluationArtifact, get_round_adapter
from src.meta_evolver.services import LocalRuntimeServices, expected_sample_ids
from src.runner.vllm_server import build_vllm_command, endpoint_is_ready
from src.utils.persistence import atomic_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EvolutionSelection:
    domain: str
    config_path: Path
    evolution_id: str
    pair_id: str
    status: str
    winner: dict[str, Any]
    registry_path: Path
    journal_path: Path


@dataclass(frozen=True)
class FullTestPlan:
    project_root: Path
    selection: EvolutionSelection
    config_path: Path
    effective_config: dict[str, Any]
    runtime_config_path: Path
    data_path: Path
    output_dir: Path
    artifact: EvaluationArtifact
    domain_python: str
    model_id: str
    model_path: str
    harness: str
    base_url: str
    max_turns: int
    concurrency: int
    vllm_settings: dict[str, Any]
    vllm_command: list[str]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _journal_path(
    project_root: Path,
    domain: str,
    evolution_id: str,
    record: dict[str, Any],
) -> Path:
    configured_value = str(record.get("journal") or "").strip()
    candidates = []
    if configured_value:
        configured = Path(configured_value).expanduser()
        candidates.append(
            configured if configured.is_absolute() else project_root / configured
        )
    candidates.extend(
        (
            project_root / "output" / domain / "rounds" / evolution_id / "journal.json",
            project_root
            / "output"
            / "meta_evolver"
            / domain
            / "rounds"
            / evolution_id
            / "journal.json",
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"journal for {evolution_id} was not found; tried: {tried}")


def _winner(record: dict[str, Any], journal_path: Path) -> dict[str, Any]:
    pair_id = str(record.get("winner_pair_id") or "")
    if not pair_id:
        raise ValueError(f"round has no winner_pair_id: {journal_path}")
    outcome = _read_json(journal_path).get("outcome") or {}
    if not isinstance(outcome, dict):
        raise ValueError(f"journal outcome is malformed: {journal_path}")
    candidates = [outcome.get("winner"), outcome.get("selected")]
    winner = next(
        (
            candidate
            for candidate in candidates
            if isinstance(candidate, dict) and str(candidate.get("id")) == pair_id
        ),
        None,
    )
    if winner is None:
        raise ValueError(
            f"winner pair {pair_id} is absent from journal outcome: {journal_path}"
        )
    model = winner.get("model")
    if not winner.get("harness") or not isinstance(model, dict):
        raise ValueError(f"winner pair is missing harness/model metadata: {journal_path}")
    if not model.get("id") or not model.get("path"):
        raise ValueError(f"winner model is missing id/path metadata: {journal_path}")
    return winner


def _domain_sources(project_root: Path) -> list[tuple[str, Path, Path]]:
    """Discover configured domains and derive their LINEAGE registries."""

    sources: list[tuple[str, Path, Path]] = []
    for config_path in sorted((project_root / "configs").glob("*.yaml")):
        config = load_config(config_path)
        domain = str((config.get("project") or {}).get("domain") or "").strip()
        registry_value = str(
            (config.get("meta_evolver") or {}).get("registry_path") or ""
        ).strip()
        if not domain or not registry_value:
            continue
        registry_path = Path(registry_value).expanduser()
        if not registry_path.is_absolute():
            registry_path = project_root / registry_path
        explicit_lineage = (config.get("meta_evolver") or {}).get("lineage_registry_path")
        if explicit_lineage:
            registry_path = Path(str(explicit_lineage)).expanduser()
            if not registry_path.is_absolute():
                registry_path = project_root / registry_path
        elif registry_path.suffix:
            registry_path = registry_path.with_name(
                f"{registry_path.stem}.lineage{registry_path.suffix}"
            )
        else:
            registry_path = registry_path.with_name(registry_path.name + ".lineage.json")
        sources.append((domain, config_path.resolve(), registry_path.resolve()))
    return sources


def discover_evolution(project_root: Path, requested_id: str) -> EvolutionSelection:
    """Resolve a unique round ID, or a unique winning pair ID, across domains."""

    matches: list[tuple[str, Path, str, dict[str, Any], Path]] = []
    for domain, config_path, registry_path in _domain_sources(project_root):
        if not registry_path.is_file():
            continue
        registry = _read_json(registry_path)
        rounds = registry.get("rounds") or {}
        if not isinstance(rounds, dict):
            raise ValueError(f"registry rounds must be a mapping: {registry_path}")
        for evolution_id, record in rounds.items():
            if not isinstance(record, dict):
                continue
            if requested_id in {str(evolution_id), str(record.get("winner_pair_id") or "")}:
                matches.append(
                    (domain, config_path, str(evolution_id), record, registry_path)
                )
    if not matches:
        raise ValueError(
            f"evolution/pair ID {requested_id!r} was not found in any lineage registry"
        )
    if len(matches) > 1:
        choices = ", ".join(
            f"{domain}:{round_id}" for domain, _, round_id, _, _ in matches
        )
        raise ValueError(
            f"ID {requested_id!r} is ambiguous ({choices}); pass the complete round ID"
        )
    domain, config_path, evolution_id, record, registry_path = matches[0]
    journal_path = _journal_path(project_root, domain, evolution_id, record)
    winner = _winner(record, journal_path)
    return EvolutionSelection(
        domain=domain,
        config_path=config_path,
        evolution_id=evolution_id,
        pair_id=str(record["winner_pair_id"]),
        status=str(record.get("status") or "unknown"),
        winner=winner,
        registry_path=registry_path,
        journal_path=journal_path,
    )


def build_plan(
    project_root: Path,
    requested_id: str,
    *,
    orchestrator_python: str,
    base_model: bool = False,
) -> FullTestPlan:
    selection = discover_evolution(project_root, requested_id)
    config_path = selection.config_path
    config = load_config(config_path)
    harness = str(selection.winner["harness"])
    spec = harness_spec(config, project_root=project_root, require_active=False)
    harness_path = spec.path_for(harness)
    if not harness_path.is_dir():
        raise FileNotFoundError(f"winner harness directory does not exist: {harness_path}")
    effective_config = apply_harness_overrides(config, harness_path)

    model = selection.winner["model"]
    if base_model:
        model_config = effective_config.get("model") or {}
        model_id = str(model_config.get("name") or "").strip()
        model_path = str(model_config.get("base") or "").strip()
        if not model_id or not model_path:
            raise ValueError(
                "--base-model requires non-empty model.name and model.base in the config"
            )
    else:
        model_id = str(model["id"])
        model_path = str(model["path"])
    local_model_path = Path(model_path).expanduser()
    if local_model_path.is_absolute() and not local_model_path.exists():
        raise FileNotFoundError(f"winner model path does not exist: {local_model_path}")

    split_value = ((effective_config.get("data") or {}).get("splits") or {}).get("test")
    if not split_value:
        raise ValueError(f"{selection.domain} config has no data.splits.test")
    services = LocalRuntimeServices(project_root, training_python=orchestrator_python)
    data_path = services.project_path(str(split_value))
    if not data_path.is_file():
        raise FileNotFoundError(f"full test split does not exist: {data_path}")

    adapter = get_round_adapter(effective_config)
    domain_python = adapter.python_executable(
        effective_config, project_root, orchestrator_python
    )
    output_dir = (
        resolve_output_root(effective_config, project_root)
        / "full_test"
        / selection.evolution_id
        / harness
    )
    if base_model:
        output_dir /= "base_model"
    # Who&When uses stable stage names to choose artifact filenames; the other
    # adapters use the directory layout and accept the same common slot name.
    artifact = adapter.external_artifact(output_dir, "policy_test")
    runtime_config_path = output_dir / "resolved_config.yaml"

    vllm_settings = dict(effective_config.get("vllm") or {})
    model_config = effective_config.get("model") or {}
    vllm_settings.setdefault("max_model_len", model_config.get("max_model_len", 32768))
    host = str(vllm_settings.get("host") or "127.0.0.1")
    port = int(vllm_settings.get("port", 8000))
    base_url = f"http://{host}:{port}/v1"
    rollout = effective_config.get("rollout") or {}
    harness_config = effective_config.get("harness") or {}
    concurrency = int(rollout.get("concurrency", 1))
    max_turns = int(harness_config.get("max_turns", 10))
    if concurrency < 1 or max_turns < 1:
        raise ValueError("rollout concurrency and harness max_turns must be positive")
    vllm_command = build_vllm_command(
        settings=vllm_settings,
        model=model_path,
        served_model_name=model_id,
    )
    return FullTestPlan(
        project_root=project_root,
        selection=selection,
        config_path=config_path,
        effective_config=effective_config,
        runtime_config_path=runtime_config_path,
        data_path=data_path,
        output_dir=output_dir,
        artifact=artifact,
        domain_python=domain_python,
        model_id=model_id,
        model_path=model_path,
        harness=harness,
        base_url=base_url,
        max_turns=max_turns,
        concurrency=concurrency,
        vllm_settings=vllm_settings,
        vllm_command=vllm_command,
    )


def _evaluation_commands(plan: FullTestPlan, *, resume: bool) -> list[list[str]]:
    adapter = get_round_adapter(plan.effective_config)
    return [
        command
        for command, _ in adapter.evaluation_commands(
            python=plan.domain_python,
            config=plan.runtime_config_path,
            data=plan.data_path,
            split="test",
            model=plan.model_id,
            base_url=plan.base_url,
            harness=plan.harness,
            max_turns=plan.max_turns,
            concurrency=plan.concurrency,
            artifact=plan.artifact,
            limit=None,
            resume=resume,
        )
    ]


def _print_plan(plan: FullTestPlan) -> None:
    print("CoEvoX evolved-pair full test")
    print(f"domain={plan.selection.domain}")
    print(f"evolution_id={plan.selection.evolution_id}")
    print(f"round_status={plan.selection.status}")
    print(f"pair_id={plan.selection.pair_id}")
    print(f"harness={plan.harness}")
    print(f"model_id={plan.model_id}")
    print(f"model_path={plan.model_path}")
    print(f"test_data={plan.data_path}")
    print(f"output={plan.output_dir}")


def execute_plan(
    plan: FullTestPlan, *, dry_run: bool, resume: bool, base_model: bool = False
) -> int:
    _print_plan(plan)
    commands = _evaluation_commands(plan, resume=resume)
    if dry_run:
        print("vllm_command=" + shlex.join(plan.vllm_command))
        for index, command in enumerate(commands, start=1):
            print(f"evaluation_command_{index}=" + shlex.join(command))
        return 0

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(plan.effective_config, plan.runtime_config_path)
    atomic_json(
        # HomeBench owns run_manifest.json for strict rollout provenance.
        plan.output_dir / "evolution_full_test.json",
        {
            "domain": plan.selection.domain,
            "evolution_id": plan.selection.evolution_id,
            "round_status": plan.selection.status,
            "pair_id": plan.selection.pair_id,
            "harness": plan.harness,
            "model": {"id": plan.model_id, "path": plan.model_path},
            "model_source": "base" if base_model else "winner",
            "test_data": str(plan.data_path),
            "registry": str(plan.selection.registry_path),
            "journal": str(plan.selection.journal_path),
        },
    )

    adapter = get_round_adapter(plan.effective_config)
    expected = len(expected_sample_ids(plan.data_path))
    if resume:
        reusable, reason = adapter.validate_reuse(
            plan.artifact,
            expected,
            config=plan.effective_config,
            data=plan.data_path,
            limit=None,
            harness=plan.harness,
            model=plan.model_id,
            base_url=plan.base_url,
            max_turns=plan.max_turns,
        )
        if reusable:
            result = adapter.load_result(plan.artifact)
            print(f"resume=complete ({reason})")
            print("RESULT " + json.dumps(result.metrics, ensure_ascii=False, sort_keys=True))
            return 0
        if adapter.artifact_complete(plan.artifact, expected):
            raise ValueError(
                f"existing complete artifact failed identity validation ({reason}); "
                f"move {plan.output_dir} aside before rerunning"
            )

    services = LocalRuntimeServices(plan.project_root, training_python=sys.executable)
    server = None
    reused_server = endpoint_is_ready(plan.base_url, plan.model_id)
    if reused_server:
        print(f"vllm=reusing matching server at {plan.base_url}")
    else:
        print("vllm=starting managed server (other local vLLM processes are cleaned up)")
        server = services.start_vllm(
            plan.vllm_command,
            label="full_test",
            round_dir=plan.output_dir,
            base_url=plan.base_url,
            expected_model=plan.model_id,
            settings=plan.vllm_settings,
            dry_run=False,
        )
    try:
        services.evaluate_with_adapter(
            adapter,
            python=plan.domain_python,
            config=plan.runtime_config_path,
            data=plan.data_path,
            split="test",
            model=plan.model_id,
            base_url=plan.base_url,
            harness=plan.harness,
            max_turns=plan.max_turns,
            concurrency=plan.concurrency,
            artifact=plan.artifact,
            limit=None,
            resume=resume,
            stage="full_test",
            dry_run=False,
        )
    finally:
        if not reused_server:
            services.stop_vllm(server, label="full_test", dry_run=False)

    reusable, reason = adapter.validate_reuse(
        plan.artifact,
        expected,
        config=plan.effective_config,
        data=plan.data_path,
        limit=None,
        harness=plan.harness,
        model=plan.model_id,
        base_url=plan.base_url,
        max_turns=plan.max_turns,
    )
    if not reusable:
        raise RuntimeError(f"full-test output failed validation: {reason}")
    result = adapter.load_result(plan.artifact)
    print("RESULT " + json.dumps(result.metrics, ensure_ascii=False, sort_keys=True))
    print(f"metrics={plan.artifact.metrics}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evolution_id", help="Complete round ID, or a unique winner pair ID")
    parser.add_argument(
        "--dry-run", action="store_true", help="Resolve and print commands without using GPUs"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not pass resume mode to an incomplete evaluation",
    )
    parser.add_argument(
        "--base-model",
        action="store_true",
        help=(
            "Evaluate the selected round's evolved harness with model.name/model.base "
            "from its config instead of the round's trained winner model"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = build_plan(
            PROJECT_ROOT,
            args.evolution_id,
            orchestrator_python=sys.executable,
            base_model=args.base_model,
        )
        return execute_plan(
            plan,
            dry_run=args.dry_run,
            resume=not args.no_resume,
            base_model=args.base_model,
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"full-test subprocess failed with exit code {exc.returncode}: {exc.cmd}",
            file=sys.stderr,
        )
        return int(exc.returncode or 1)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
