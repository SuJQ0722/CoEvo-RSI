"""LINEAGE ports backed by existing domain adapters, MetaAgent CLI and SFT tools."""

from __future__ import annotations

import copy
import fcntl
import json
import logging
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

from src.config import (
    apply_harness_overrides,
    harness_spec,
    redact_config_secrets,
    set_active_harness,
    write_resolved_config,
)
from src.domains import get_domain_plugin
from src.domains.io import read_records
from src.domains.round_adapter import EvaluationArtifact, get_round_adapter
from src.meta_evolver.batch_summary import summarize_failure_batch
from src.meta_evolver.context import (
    curate_context,
    generate_briefs,
    parse_json_response,
    run_meta_text,
)
from src.meta_evolver.costs import ledger_summary, meta_usage
from src.meta_evolver.evolution import (
    load_harness_tree,
    publish_lineage_harnesses,
    semantic_harness_hash,
)
from src.meta_evolver.inheritance import collect_harnesses, inheritance_settings
from src.meta_evolver.lineage import (
    Journal,
    candidate_name_for_slot,
    digest,
    evolve_round,
    quality_names,
    resume_config,
    select_model_pairs,
    tolerance,
)
from src.meta_evolver.merging import merge_models, merge_recipe
from src.meta_evolver.runner import MetaEvolverError, run_meta_evolver
from src.meta_evolver.services import RuntimeServices
from src.meta_evolver.weights import (
    compatible,
    describe_model,
    file_hash,
    harness_hash,
    merge_half,
    merge_reason,
    tree_hash,
)
from src.runner.vllm_server import build_vllm_command
from src.training.ms_swift import (
    build_ms_swift_sft_command,
    resumable_ms_swift_checkpoint,
)
from src.utils.persistence import atomic_json

LOGGER = logging.getLogger("pipeline.lineage")


def _records(path):
    return list(read_records(path))


def validate_splits(train, val, test):
    """Freeze split bytes and keep every training example outside evaluation."""
    paths = {"train": train.resolve(), "val": val.resolve(), "test": test.resolve()}
    if paths["train"] in {paths["val"], paths["test"]}:
        raise ValueError("LINEAGE requires the train path to be separate from evaluation")
    identities, sets = {}, {}
    for split, path in paths.items():
        ids = [
            str(r.get("sample_id") or r.get("source_id") or r.get("id") or "")
            for r in read_records(path)
        ]
        if not ids or not all(ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{split} has missing or duplicate sample IDs, or is empty")
        identities[split] = {"path": str(path), "sha256": file_hash(path), "task_ids": ids}
        sets[split] = set(ids)
    for a, b in (("train", "val"), ("train", "test")):
        if sets[a] & sets[b]:
            raise ValueError(f"{a}/{b} splits overlap by sample ID")
    return identities


def execution_tokens(rollouts: Path):
    """Count recorded model usage; incomplete usage remains unknown, never zero."""
    total = 0
    records = _records(rollouts)
    for record in records:
        if record.get("token_usage_available") is False:
            return None
        usage = record.get("token_usage") or record.get("usage") or {}
        count = usage.get("total_tokens") if isinstance(usage, dict) else None
        if count is None:
            count = record.get("total_tokens")
        if count is None and isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
            count = float(usage["prompt_tokens"]) + float(usage.get("completion_tokens", 0))
        if count is None:
            # Deterministic RBAC rule routes legitimately make no LLM request.
            route = record.get("route") or record.get("routing")
            if route in ("rule", "rule_allow", "rule_deny"):
                count = 0
            else:
                return None
        total += float(count)
    return total if records else None


def _source_packet(path):
    return {
        p.relative_to(path).as_posix(): p.read_text(errors="replace")
        for p in sorted(path.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix in {".py", ".yaml", ".json", ".sh", ".txt"}
    }


def _clean_historical_record(record):
    # Historical validation/test details are controller-only even after publish.
    allowed = (
        "id",
        "parent",
        "harness",
        "source_hash",
        "model_ids",
        "accepted",
        "status",
        "adopted_designs",
        "rejected_designs",
        "dependencies",
        "reference_ids",
        "scores",
    )
    return {key: record[key] for key in allowed if key in record}


class Runtime:
    def __init__(
        self,
        *,
        args,
        config,
        config_path,
        project_root,
        output_root,
        round_dir,
        domain_python,
        model_source,
        base_url,
        vllm_settings,
        training_config,
        services: RuntimeServices,
        train_data,
        val_data,
        test_data,
    ):
        self.args, self.config, self.config_path = args, copy.deepcopy(config), config_path
        self.root, self.output, self.round_dir = project_root, output_root, round_dir
        self.py, self.services = domain_python, services
        self.base_url, self.vllm_settings = base_url, vllm_settings
        self.training_config = training_config
        self.train_data, self.val_data, self.test_data = train_data, val_data, test_data
        self.adapter, self.plugin = get_round_adapter(config), get_domain_plugin(config)
        self.spec = harness_spec(config, project_root=project_root)
        self.round_id = round_dir.name
        self.meta = self.config["meta_evolver"]
        self.selection = self.config["selection"]
        self.inheritance = inheritance_settings(self.config)
        self.model_source = model_source
        self.cache_root = output_root / "lineage"
        self.weights_root = self.cache_root / "models"
        self.registry_path = services.project_path(self.meta["registry_path"])
        self.archive_path = self.registry_path.with_suffix(".lineage.json")
        self.server = None
        self.loaded_model = None
        self.server_started = None
        self.journal = None
        self.published = False
        resume_validation = dict(self.config.get("resume_validation") or {})
        self.validate_resume_data = bool(resume_validation.get("data", True))
        self.validate_resume_model = bool(resume_validation.get("model", True))
        self.validate_resume_harness = bool(resume_validation.get("harness", True))

    def initialize(self):
        initialize_start = time.monotonic()
        legacy_state = self.round_dir / "round_state.json"
        if legacy_state.exists() and not (self.round_dir / "snapshot.json").exists():
            raise ValueError(
                "legacy round has no frozen LINEAGE state; start a new round to import its historical outputs"
            )
        self.round_dir.mkdir(parents=True, exist_ok=True)
        self.split_identity = validate_splits(self.train_data, self.val_data, self.test_data)
        dependency_reader = getattr(self.adapter, "evaluation_dependencies", lambda *args: {})
        self.dependencies = {
            split: dependency_reader(self.config, path, self.root)
            for split, path in (("train", self.train_data), ("val", self.val_data))
        }
        self.archive = (
            json.loads(self.archive_path.read_text())
            if self.archive_path.exists()
            else {"version": 1, "models": {}, "harnesses": {}, "rounds": {}, "incumbent": None}
        )
        frozen_path = self.round_dir / "snapshot.json"
        if frozen_path.exists():
            self.snapshot = json.loads(frozen_path.read_text())
        else:
            self.snapshot = copy.deepcopy(self.archive)
            self.snapshot["legacy_harness_tree"] = load_harness_tree(self.registry_path)
            # Import historical SFT outputs once at the round boundary, not during traversal.
            old_paths = []
            rounds = self.output / "rounds"
            for old_round in sorted(rounds.iterdir()) if rounds.exists() else []:
                if old_round == self.round_dir or not old_round.is_dir():
                    continue
                for checkpoint in sorted(
                    (old_round / "training").glob("**/checkpoint-*/config.json")
                ):
                    old_paths.append(str(checkpoint.parent.resolve()))
            self.snapshot["legacy_model_paths"] = old_paths
            self.snapshot["legacy_parent_sources"] = {}
            for source in old_paths:
                old_round = Path(source).relative_to(rounds).parts[0]
                state_path = rounds / old_round / "round_state.json"
                if state_path.exists():
                    state = json.loads(state_path.read_text())
                    self.snapshot["legacy_parent_sources"][source] = state.get("model_path")
            self.snapshot["legacy_weight_signatures"] = {
                source: {
                    str(p): [p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns]
                    for p in sorted(Path(source).iterdir())
                    if p.is_file()
                    and p.suffix in {".safetensors", ".bin", ".json", ".model", ".txt"}
                }
                for source in old_paths
            }
            self.snapshot["history_packets"] = []
            nodes = {
                **self.snapshot["legacy_harness_tree"].get("nodes", {}),
                **self.snapshot["harnesses"],
            }
            for name, node in nodes.items():
                try:
                    source = self.spec.path_for(name)
                    if source.is_dir():
                        self.snapshot["history_packets"].append(
                            {
                                "id": name,
                                **_clean_historical_record(node),
                                "source": _source_packet(source),
                            }
                        )
                except ValueError:
                    continue
            self.snapshot["id"] = "archive_" + digest(self.snapshot)
            atomic_json(frozen_path, self.snapshot)
        incumbent = self.snapshot.get("incumbent")
        if incumbent:
            self.parent = incumbent["harness"]
            self.model_source = incumbent["model"]["path"]
            self.args.model = incumbent.get("served_model", incumbent["model"]["id"])
        else:
            self.parent = self.spec.active
        self.config["harness"]["active"] = self.parent
        self.config["harness"]["root"] = str(self.spec.root)
        self.config.setdefault("rollout", {}).setdefault("seed", 0)
        self.parent_source = self.spec.path_for(self.parent)
        frozen_source = self.round_dir / "incumbent_source"
        if not frozen_source.exists():
            shutil.copytree(
                self.parent_source,
                frozen_source,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        self.frozen_source = frozen_source
        if self.validate_resume_harness and tree_hash(frozen_source) != tree_hash(
            self.parent_source
        ):
            raise ValueError("incumbent harness changed since the round was frozen")
        if not self.validate_resume_harness:
            # Explicit opt-out means current source wins over the frozen copy.
            # Keep the old copy intact for auditability, but seed/evaluate from
            # the live harness selected by the lineage incumbent.
            self.frozen_source = self.parent_source
        self.parent_hash = harness_hash(frozen_source)
        base_source = (self.config.get("model") or {}).get("base", self.model_source)
        base_error = None
        try:
            self.base = describe_model(
                base_source, project_root=self.root, cache_root=self.weights_root, route="base"
            )
            self.base["common_base_id"] = self.base["effective_weights_hash"]
        except (OSError, ValueError) as exc:
            self.base, base_error = None, str(exc)
        common = incumbent["model"].get("common_base_id") if incumbent else None
        self.latest = describe_model(
            self.model_source,
            project_root=self.root,
            cache_root=self.weights_root,
            common_base_id=common,
            route="incumbent",
        )
        if incumbent:
            self.latest.update(
                {
                    key: value
                    for key, value in incumbent["model"].items()
                    if key
                    in {"parent_ids", "parent_weights", "route", "training_id", "merge_recipe"}
                }
            )
        if (
            incumbent
            and self.validate_resume_model
            and self.latest["effective_weights_hash"]
            != incumbent["model"]["effective_weights_hash"]
        ):
            raise ValueError("incumbent model weights changed after acceptance")
        if self.base and self.latest["id"] == self.base["id"]:
            self.latest["common_base_id"] = self.base["common_base_id"]
        elif not incumbent:
            stored = self.snapshot["models"].get(self.latest["id"], {})
            self.latest["common_base_id"] = stored.get("common_base_id")
        # Runtime source is a durable model identity, never just the served alias.
        self.config.setdefault("model", {})["revision"] = self.latest["effective_weights_hash"]
        self.code_hash = digest(
            {
                "core": tree_hash(self.root / "src", suffixes={".py"}),
                "harness_package": {
                    p.name: file_hash(p)
                    for p in self.spec.root.iterdir()
                    if p.is_file() and p.suffix in {".py", ".json", ".yaml"}
                },
            }
        )
        from importlib.metadata import distributions

        self.environment_identity = {
            "packages": {dist.metadata["Name"]: dist.version for dist in distributions()},
            "domain_python": str(self.py),
        }
        effective = redact_config_secrets(self.config)
        identity = {
            "round_id": self.round_id,
            "round_number": self.args.round_number,
            "snapshot_id": self.snapshot["id"],
            "config": effective,
            "splits": self.split_identity,
            "dependencies": self.dependencies,
            "model_hash": self.latest["effective_weights_hash"],
            "harness_hash": self.parent_hash,
            "code_hash": self.code_hash,
            "environment_version": self.environment_identity,
        }
        if getattr(self.args, "stage", "evolve") == "model-selection":
            identity["stage"] = "model-selection"
            identity["selection_harnesses"] = {
                name: harness_hash(self.spec.path_for(name)) for name in self.args.selection_harness
            }
        ignored_identity_fields = set()
        if not self.validate_resume_data:
            ignored_identity_fields.update({"splits", "dependencies"})
        if not self.validate_resume_model:
            ignored_identity_fields.add("model_hash")
        if not self.validate_resume_harness:
            ignored_identity_fields.add("harness_hash")
        self.journal = Journal(
            self.round_dir / "journal.json",
            identity,
            allow_code_change=bool(getattr(self.args, "resume_parent_artifacts", False)),
            ignored_identity_fields=ignored_identity_fields,
        )
        self.journal.data.setdefault("costs", [])
        self.journal.data.setdefault("model_exclusions", [])
        if base_error:
            self.exclude(str(base_source), base_error)
        self.journal.save()
        self.cost("freeze_and_identify", initialize_start, gpu_hours=0)
        write_resolved_config(self.config, self.round_dir / "effective_config.yaml")
        atomic_json(
            self.round_dir / "round_state.json",
            {
                "round_number": self.args.round_number,
                "candidate": self.round_id,
                "model": self.args.model,
                "model_path": str(self.model_source),
                "parent_harness": self.parent,
                "snapshot_id": self.snapshot["id"],
            },
        )

    def select_models(self):
        """Standalone full-val selection, isolated from round publication and SFT."""
        incumbent = self.journal.once("incumbent_eval", self.incumbent_eval)
        children = []
        for name in self.args.selection_harness:
            report = self.plugin.smoke_test(name, package=self.spec.package, root=self.spec.root)
            if not report.passed:
                raise ValueError(f"selection harness failed smoke checks: {name}: {report}")
            pair = self.evaluate({"harness": name}, self.latest)
            if pair.get("evaluable", True):
                children.append(pair)
        ranking = select_model_pairs(self, self.journal, children, incumbent, self.selection)
        from src.diagnostics.mismatch import round_comparisons

        report = {
            "stage": "model-selection",
            "domain": self.adapter.name,
            "ranking": ranking,
            "pair_pool": self.journal.data.get("pair_pool", []),
            "model_exclusions": self.journal.data.get("model_exclusions", []),
            "mismatch": round_comparisons(self.journal.data),
            "inheritance": self.inheritance_report(),
            "published": False,
        }
        atomic_json(self.round_dir / "model_selection.json", report)
        return report

    def cost(self, stage, start, **values):
        self.journal.data["costs"].append(
            {
                "stage": stage,
                "wall_seconds": time.monotonic() - start,
                "gpu_hours": values.pop("gpu_hours", None),
                "api_cost": values.pop("api_cost", None),
                **values,
            }
        )
        self.journal.save()

    def exclude(self, model, reason):
        record = {"model": model, "reason": reason}
        if record not in self.journal.data["model_exclusions"]:
            self.journal.data["model_exclusions"].append(record)
            self.journal.save()

    def account_serving(self):
        if self.server_started is not None and self.journal is not None:
            workers = int(self.vllm_settings.get("tensor_parallel_size", 1)) * int(
                self.vllm_settings.get("data_parallel_size", 1)
            )
            start = self.server_started
            self.server_started = time.monotonic()
            self.cost(
                "model_residency",
                start,
                model_id=self.loaded_model,
                gpu_hours=(self.server_started - start) * workers / 3600,
                gpu_hours_basis="allocated_workers_times_wall",
            )

    def stop(self):
        self.account_serving()
        self.server_started = None
        if self.server:
            self.services.stop_vllm(self.server, label="lineage-model-switch", dry_run=False)
            self.server = None
        self.loaded_model = None

    def ensure_model(self, model):
        if self.loaded_model == model["id"]:
            return
        for filename, signature in model["file_signature"].items():
            stat = Path(filename).stat()
            if [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns] != signature:
                # Copying/restoring a checkpoint changes ctime without changing
                # its content. Revalidate content before refreshing metadata.
                current = describe_model(
                    model["path"], project_root=self.root, cache_root=self.weights_root
                )
                for key in (
                    "effective_weights_hash", "architecture_hash", "tokenizer_hash",
                    "layout_hash", "precision",
                ):
                    if current[key] != model[key]:
                        raise ValueError(
                            f"model content changed after snapshot ({key}): {model['id']}"
                        )
                model["file_signature"] = current["file_signature"]
                break
        self.stop()
        start = time.monotonic()
        self.server = self.services.start_vllm(
            build_vllm_command(
                settings=self.vllm_settings, model=model["path"], served_model_name=model["id"]
            ),
            label="policy",
            round_dir=self.round_dir,
            base_url=self.base_url,
            expected_model=model["id"],
            settings=self.vllm_settings,
            dry_run=False,
        )
        self.loaded_model = model["id"]
        self.server_started = start
        self.cost("model_load", start, model_id=model["id"])

    def incumbent_eval(self):
        return self.evaluate(
            {"harness": self.parent, "harness_hash": self.parent_hash}, self.latest
        )

    def execution_config(self, directory, model):
        effective = copy.deepcopy(self.config)
        effective["model"]["revision"] = model["effective_weights_hash"]

        # Services load a snapshot with exactly the resolved runtime overrides.
        # Credential values remain environment references, not persisted secrets.
        def protect(item, prefix="LINEAGE"):
            if isinstance(item, dict):
                result = {}
                for key, value in item.items():
                    name = prefix + "_" + key.upper()
                    if (
                        key.casefold()
                        in {"api_key", "apikey", "access_token", "password", "secret", "token"}
                        and value
                    ):
                        os.environ[name] = str(value)
                        result[key] = "${oc.env:" + name + "}"
                    else:
                        result[key] = protect(value, name)
                return result
            if isinstance(item, list):
                return [protect(value, prefix + "_" + str(i)) for i, value in enumerate(item)]
            return item

        import yaml

        path = directory / "runtime.yaml"
        path.write_text(yaml.safe_dump(protect(effective), allow_unicode=True, sort_keys=False))
        return path

    def evaluate(self, child, model, *, split="val"):
        data_path = self.val_data if split == "val" else self.train_data
        dependency_reader = getattr(self.adapter, "evaluation_dependencies", lambda *args: {})
        if (
            self.validate_resume_data
            and dependency_reader(self.config, data_path, self.root) != self.dependencies[split]
        ):
            raise ValueError("environment inputs changed after the round was frozen")
        if self.validate_resume_data and any(
            file_hash(path) != self.split_identity[name]["sha256"]
            for name, path in (("train", self.train_data), ("val", self.val_data))
        ):
            raise ValueError("dataset changed after the round was frozen")
        harness = child.get("harness") or child.get("candidate")
        source = self.spec.path_for(harness)
        hhash = harness_hash(source)
        if child.get("harness_hash") and hhash != child["harness_hash"]:
            raise ValueError("harness source changed after candidate generation")
        effective = apply_harness_overrides(self.config, source)
        identity = {
            "model": model["effective_weights_hash"],
            "split": split,
            "harness": hhash,
            "tasks": self.split_identity[split],
            "train_identity": self.split_identity["train"],
            "environment_inputs": self.dependencies[split],
            "environment": effective.get("execution", {}),
            "environment_version": self.environment_identity,
            "tool_schema_version": self.code_hash,
            "evaluator_version": self.code_hash,
            "decoding": effective.get("rollout", {}),
            "seed": effective.get("rollout", {}).get("seed"),
            "initial_memory": "fresh_per_task",
            "config": redact_config_secrets(effective),
            "max_turns": self.args.max_turns,
        }
        # API credentials do not affect inference semantics.
        identity["config"].pop("meta_evolver", None)
        identity["config"].get("harness", {}).pop("active", None)
        identity["config"].get("model", {}).pop("revision", None)
        key = digest(identity)
        cache = self.cache_root / "evaluations" / key
        cache.mkdir(parents=True, exist_ok=True)
        manifest = cache / "identity.json"
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError("evaluation cache identity mismatch")
        atomic_json(manifest, identity)
        artifact = self.adapter.artifact(cache, "child_val")
        expected = len(self.split_identity[split]["task_ids"])
        cached = self.adapter.artifact_complete(artifact, expected)
        start = time.monotonic()
        result = None
        try:
            if not cached:
                self.ensure_model(model)
                result = self.services.evaluate_with_adapter(
                    self.adapter,
                    python=self.py,
                    config=self.execution_config(cache, model),
                    data=data_path,
                    split=split,
                    model=model["id"],
                    base_url=self.base_url,
                    harness=harness,
                    max_turns=self.args.max_turns,
                    concurrency=int(self.config["rollout"]["concurrency"]),
                    artifact=artifact,
                    limit=None,
                    resume=bool(getattr(self.args, "resume_parent_artifacts", False)),
                    stage=f"lineage.{split}.{harness}",
                    dry_run=False,
                )
            else:
                result = self.adapter.load_result(artifact)
            if not self.adapter.artifact_complete(artifact, expected):
                raise RuntimeError("evaluation did not produce the full task set")
            actual_ids = {str(r["sample_id"]) for r in _records(artifact.rollouts)}
            if actual_ids != set(self.split_identity[split]["task_ids"]):
                raise ValueError("evaluation IDs differ from the frozen full split")
            if not all(
                math.isfinite(float(result.metrics[name])) for name in quality_names(self.selection)
            ):
                raise ValueError("non-finite validation score")
        except (subprocess.CalledProcessError, OSError, RuntimeError) as exc:
            diagnostics = self.mismatch_diagnostics(artifact, identity, result)
            self.cost(
                "evaluation_failed", start, model_id=model["id"], harness=harness, reason=str(exc)
            )
            if harness == self.parent and model["id"] == self.latest["id"]:
                raise
            self.exclude(model["id"], f"{harness}: {exc}")
            return {
                "id": "p_" + key,
                "evaluable": False,
                "harness": harness,
                "harness_hash": hhash,
                "model": model,
                "reason": str(exc),
                "diagnostics": {"mismatch": diagnostics},
            }
        pair = {
            "id": "p_" + key,
            "eval_id": key,
            "harness": harness,
            "harness_hash": hhash,
            "model": model,
            "scores": result.metrics,
            "quality_score": list(self.adapter.quality_score(result.metrics, self.selection)),
            "token_cost": execution_tokens(artifact.rollouts),
            "artifact": {
                "root": str(artifact.root),
                "rollouts": str(artifact.rollouts),
                "metrics": str(artifact.metrics),
            },
            "phase": "pre",
            "evaluable": True,
            "split": split,
            "task_scope": "full",
            "diagnostics": {"mismatch": self.mismatch_diagnostics(artifact, identity, result)},
        }
        if key not in self.journal.data["evaluations"]:
            self.journal.data["evaluations"][key] = pair
            self.cost(
                "evaluation",
                start,
                split=split,
                model_id=model["id"],
                harness=harness,
                token_cost=pair["token_cost"],
                cache_hit=cached,
                task_count=expected,
                new_rollouts=0 if cached else expected,
            )
        else:
            self.journal.data["evaluations"][key]["diagnostics"] = pair["diagnostics"]
        return pair

    def mismatch_diagnostics(self, artifact, identity, result=None):
        from src.diagnostics.mismatch import MismatchContract, write_report

        try:
            contract = getattr(self.adapter, "mismatch_contract", MismatchContract)()
            return write_report(
                artifact.rollouts,
                contract,
                identity={
                    "pair_id": "p_" + digest(identity),
                    "model_weights_hash": identity["model"],
                    "harness_hash": identity["harness"],
                    "split": identity["split"],
                    "tasks": identity["tasks"],
                    "decoding": identity["decoding"],
                },
                per_sample=result.per_sample if result else (),
            )
        except Exception as exc:  # noqa: BLE001 -- diagnostics cannot change selection behavior
            LOGGER.warning("mismatch diagnostics unavailable for %s: %s", artifact.root, exc)
            return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

    def _result(self, pair):
        return self.adapter.load_result(
            EvaluationArtifact(**{k: Path(v) for k, v in pair["artifact"].items()})
        )

    def admit(self, incumbent, pair):
        if not pair.get("evaluable", True):
            return False, pair.get("reason", "unevaluable_pair")
        # Who&When retains its existing tolerance-only harness prefilter.
        if not self.adapter.prefilter_requires_improvement:
            epsilon = float(self.selection.get("epsilon", 1e-12))
            ok = all(
                pair["scores"][name]
                >= incumbent["scores"][name] - tolerance(self.selection, name) - epsilon
                for name in quality_names(self.selection)
            )
            return ok, "within_parent_child_tolerances" if ok else "quality_regression"
        return self.accept(incumbent, pair)

    def accept(self, incumbent, pair):
        if not pair.get("evaluable", True):
            return False, "unevaluable_pair"
        decision = self.plugin.select(self._result(incumbent), self._result(pair), self.selection)
        return decision.accepted, decision.reason

    def curate(self):
        # The incumbent's full validation result is created immediately before
        # curation by evolve_round. Reuse it as the sole source of failure
        # evidence; the training split is reserved for rejection sampling.
        val_pair = self.journal.data.get("incumbent_eval")
        if val_pair is None:
            val_pair = self.incumbent_eval()
        if val_pair.get("split") != "val":
            raise ValueError("Curator evidence must come from the validation split")
        evidence_dir = self.round_dir / "validation_evidence"
        evidence_dir.mkdir(exist_ok=True)
        evidence = evidence_dir / "evidence.jsonl"
        self.plugin.build_evidence(Path(val_pair["artifact"]["rollouts"]), evidence)
        inputs_path = self.round_dir / "curator_inputs.json"
        if inputs_path.exists():
            inputs = json.loads(inputs_path.read_text())
        else:
            inputs = {
                "archive_snapshot_id": self.snapshot["id"],
                "incumbent": self.parent,
                "incumbent_source": _source_packet(self.frozen_source),
                "validation_evidence": _records(evidence),
                "quality_metrics": list(quality_names(self.selection)),
                "history": [
                    card
                    for card in self.snapshot.get("history_packets", [])
                    if card["id"] != self.parent
                ],
                "domain_contract": self.plugin.meta_user_prompt(self.config),
            }
            if self.inheritance and self.inheritance["harness"]["mode"] == "predefined":
                inputs["history"], inputs["retrieval"] = collect_harnesses(
                    inputs, self.inheritance["harness"]
                )
            atomic_json(inputs_path, inputs)
        start = time.monotonic()
        packet = curate_context(
            config=self.config,
            project_root=self.root,
            output_dir=self.round_dir / "curator",
            inputs=inputs,
            backend=self.args.backend,
        )
        self.cost("curator", start, usage=meta_usage(self.round_dir / "curator" / "stdout.log"))
        return packet

    def briefs(self, initial_context, feedback, batch):
        start = time.monotonic()
        context, briefs = generate_briefs(
            config=self.config,
            project_root=self.root,
            output_dir=self.round_dir / batch["id"] / "brief_generation",
            initial_context=initial_context,
            feedback=feedback,
            batch=batch,
            backend=self.args.backend,
        )
        self.cost(
            "brief_generation",
            start,
            batch_id=batch["id"],
            count=batch["count"],
            usage=meta_usage(self.round_dir / batch["id"] / "brief_generation" / "stdout.log"),
        )
        return context, briefs

    def edit(self, context, brief, slot):
        candidate = candidate_name_for_slot(self.round_id, slot)
        directory = self.round_dir / f"harness_attempt_{slot:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        receipt = directory / "proposal.json"
        if receipt.exists():
            return json.loads(receipt.read_text())
        atomic_json(directory / "batch_context.json", context)
        atomic_json(directory / "brief.json", brief)
        start = time.monotonic()
        proposal = {
            "candidate": candidate,
            "harness": candidate,
            "slot": slot,
            "brief_id": brief["id"],
            "batch_context_id": context["id"],
            "initial_context_id": context["initial_context"]["id"],
            "prior_failure_summary_ids": [x["id"] for x in context["failure_attributions"]],
            "in_batch_sibling_feedback_received": False,
            "valid": False,
        }
        try:
            result = run_meta_evolver(
                self.config,
                self.root,
                directory / "batch_context.json",
                backend=self.args.backend,
                candidate_name=candidate,
                batch_context_path=directory / "batch_context.json",
                brief_path=directory / "brief.json",
                run_dir_override=directory / "editor",
                parent_source_path=self.frozen_source,
                publish_registry=False,
            )
            source = self.spec.path_for(candidate)
            manifest = json.loads((result.run_dir / "manifest.json").read_text())
            proposal.update(
                valid=True,
                source_path=str(source),
                smoke=manifest.get("smoke"),
                harness_hash=harness_hash(source),
                semantic_hash=semantic_harness_hash(source),
                editor_session_id=str(result.run_dir),
            )
            lineage_path = source / "lineage.json"
            if not lineage_path.exists():
                raise ValueError(
                    "Editor did not record adopted references/dependencies in lineage.json"
                )
            if lineage_path.exists():
                lineage = json.loads(lineage_path.read_text())
                known = set(brief.get("reference_ids") or [])
                if not set(lineage.get("reference_ids") or []).issubset(known):
                    raise ValueError("Editor adopted a reference outside its brief")
                proposal["lineage"] = lineage
        except (MetaEvolverError, OSError, ValueError, RuntimeError) as exc:
            proposal.update(valid=False, reason=str(exc))
        proposal["wall_seconds"] = time.monotonic() - start
        atomic_json(receipt, proposal)
        # Parallel workers persist only their own receipts. Journal writes are controller-only.
        return proposal

    def summarize(self, batch):
        entries = []
        for result in batch["results"]:
            proposal = result["proposal"]
            pair = result.get("pair", {})
            entries.append(
                {
                    "candidate": proposal["candidate"],
                    "reason": result["reason"],
                    "scores": pair.get("scores"),
                    "changes": proposal.get("lineage", {}),
                    "valid": proposal.get("valid"),
                }
            )
        feedback = {
            "batch_id": batch["id"],
            "candidate_count": batch["count"],
            "quality_metrics": quality_names(self.selection),
            "incumbent_scores": self.journal.data["incumbent_eval"]["scores"],
            "candidates": entries,
        }
        feedback_path = self.round_dir / batch["id"] / "failures.json"
        atomic_json(feedback_path, feedback)
        start = time.monotonic()
        path, _ = summarize_failure_batch(
            config=self.config,
            project_root=self.root,
            feedback_path=feedback_path,
            output_dir=self.round_dir / batch["id"] / "summary",
            backend=self.args.backend,
            aggregate_feedback=feedback,
        )
        summary = {
            "id": f"failure_{batch['id']}",
            "batch_id": batch["id"],
            "text": path.read_text(),
        }
        self.cost(
            "failure_attribution",
            start,
            batch_id=batch["id"],
            usage=meta_usage(self.round_dir / batch["id"] / "summary" / "stdout.log"),
        )
        return summary

    def models(self):
        inventory_start = time.monotonic()
        models = {self.latest["effective_weights_hash"]: self.latest}
        candidates = list(self.snapshot["models"].values())
        if self.base:
            candidates.append(self.base)
        candidates.extend({"path": path} for path in self.snapshot.get("legacy_model_paths", []))
        described = {}

        def describe_ancestor(source, visited=()):
            if not source or source in visited:
                return None
            if source in described:
                return described[source]
            known = next(
                (item for item in self.snapshot["models"].values() if item["path"] == source), {}
            )
            model = describe_model(
                source,
                project_root=self.root,
                cache_root=self.weights_root,
                common_base_id=known.get("common_base_id"),
            )
            described[source] = model
            if self.base and model["id"] == self.base["id"]:
                model["common_base_id"] = self.base["common_base_id"]
            parent_source = self.snapshot.get("legacy_parent_sources", {}).get(source)
            if parent_source and not model.get("common_base_id"):
                ancestor = describe_ancestor(parent_source, (*visited, source))
                if ancestor and compatible(ancestor, model):
                    model["common_base_id"] = ancestor.get("common_base_id")
                    model["parent_ids"] = [ancestor["id"]]
            return model

        for candidate in candidates:
            source = candidate["path"]
            try:
                for filename, signature in (
                    self.snapshot.get("legacy_weight_signatures", {}).get(source, {}).items()
                ):
                    stat = Path(filename).stat()
                    if [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns] != signature:
                        raise ValueError("legacy weights changed after the archive snapshot")
                model = describe_model(
                    source,
                    project_root=self.root,
                    cache_root=self.weights_root,
                    common_base_id=candidate.get("common_base_id"),
                    parent_ids=candidate.get("parent_ids", []),
                    route=candidate.get("route", "historical"),
                )
                model.update(
                    {
                        key: candidate[key]
                        for key in ("parent_weights", "merge_recipe", "training_id")
                        if key in candidate
                    }
                )
                if not model.get("common_base_id") and source in self.snapshot.get(
                    "legacy_parent_sources", {}
                ):
                    ancestor_model = describe_ancestor(source)
                    model["common_base_id"] = ancestor_model.get("common_base_id")
                    model["parent_ids"] = ancestor_model.get("parent_ids", [])
                if (
                    candidate.get("effective_weights_hash")
                    and model["effective_weights_hash"] != candidate["effective_weights_hash"]
                ):
                    raise ValueError("archived effective weights changed")
                if not compatible(model, self.latest):
                    raise ValueError("incompatible architecture, tokenizer or tensor layout")
                existing = models.get(model["effective_weights_hash"])
                if existing and not existing.get("common_base_id") and model.get("common_base_id"):
                    existing.update(
                        common_base_id=model["common_base_id"],
                        parent_ids=model.get("parent_ids", []),
                    )
                else:
                    models.setdefault(model["effective_weights_hash"], model)
            except (OSError, ValueError) as exc:
                self.exclude(source, str(exc))
        self.cost("model_inventory", inventory_start, model_count=len(models), gpu_hours=0)
        return list(models.values())

    def merge(self, best, latest):
        key = digest([best["id"], latest["id"]])
        merges = self.journal.data.setdefault("merges", {})
        if key not in merges:
            reason = merge_reason(latest, best)
            start = time.monotonic()
            if reason:
                merges[key] = {"model": None, "reason": reason}
            else:
                self.stop()
                try:
                    model = merge_half(
                        latest, best, cache_root=self.weights_root, project_root=self.root
                    )
                    merges[key] = {"model": model, "reason": None}
                except (OSError, ValueError, RuntimeError) as exc:
                    merges[key] = {"model": None, "reason": str(exc)}
            self.cost(
                "merge", start, parents=[latest["id"], best["id"]], reason=merges[key]["reason"]
            )
        return merges[key]["model"]

    def merge_candidates(self, parents, **options):
        """Execute only a validated, content-addressed recipe chosen by the mode."""
        recipe = merge_recipe(parents, base=self.base, **options)
        key = digest(recipe)
        merges = self.journal.data.setdefault("merges", {})
        if key not in merges:
            self.stop()
            start = time.monotonic()
            try:
                model = merge_models(
                    parents,
                    base=self.base,
                    cache_root=self.weights_root,
                    project_root=self.root,
                    **options,
                )
                merges[key] = {"model": model, "reason": None, "recipe": recipe}
            except (OSError, ValueError, RuntimeError) as exc:
                merges[key] = {"model": None, "reason": str(exc), "recipe": recipe}
            self.cost("merge", start, recipe=recipe, reason=merges[key]["reason"])
        return merges[key]["model"]

    def model_inheritance_action(self, child, step, inputs, prompt):
        directory = (
            self.round_dir / "model_inheritance" / child["harness_hash"] / f"step_{step:03d}"
        )
        inputs = dict(
            inputs,
            harness_source=_source_packet(self.spec.path_for(child["harness"])),
            historical_scores=[
                {
                    key: card[key]
                    for key in (
                        "id",
                        "model_ids",
                        "scores",
                        "status",
                        "generation",
                        "accepted",
                    )
                    if key in card
                }
                for card in self.snapshot.get("history_packets", [])
                if card.get("scores")
            ],
        )
        start = time.monotonic()
        path = run_meta_text(
            config=self.config,
            project_root=self.root,
            output_dir=directory,
            inputs=inputs,
            prompt=prompt,
            backend=self.args.backend,
        )
        self.cost(
            "model_inheritance_agent",
            start,
            harness=child["harness"],
            step=step,
            usage=meta_usage(directory / "stdout.log"),
        )
        try:
            return parse_json_response(path.read_text())
        except ValueError:
            # Persist the invalid reply as one action, so the agent can correct it.
            return {"invalid_response": path.read_text()[:1000]}

    def inheritance_report(self):
        searches = (self.journal.data if self.journal else {}).get("model_inheritance", {})
        return {
            "settings": self.inheritance,
            "harness_retrieval": (self.journal.data if self.journal else {})
            .get("initial_context", {})
            .get("retrieval"),
            "searches": [
                {
                    key: state[key]
                    for key in (
                        "harness",
                        "mode",
                        "requested_k",
                        "available_donors",
                        "donor_ids",
                        "merge_status",
                        "merged_model_id",
                        "selected_id",
                        "selection_source",
                        "selection_reason",
                    )
                    if key in state
                }
                | {
                    "agent_actions": len(state.get("actions", [])),
                    "observed_pairs": len(state.get("pairs", [])),
                }
                for state in searches.values()
            ],
        }

    def train(self, pair):
        rank = pair["candidate_rank"]
        directory = self.round_dir / "training" / f"candidate_{rank}"
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / "training_state.json"
        training_id = f"{self.round_id}_candidate_{rank}"
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "training_id": training_id,
                "candidate_rank": rank,
                "sampling_pair_id": pair["id"],
                "input_model_id": pair["model"]["id"],
                "input_harness": pair["harness"],
                "config_hash": digest(resume_config(redact_config_secrets(self.config))),
                "post": None,
            }
        )
        # The selected pair is the durable training identity. Configuration
        # changes are intentionally allowed across resume; completed sampling
        # and training artifacts remain frozen in this state directory.
        if state["sampling_pair_id"] != pair["id"]:
            raise ValueError("training resume does not match the frozen pair")
        if state.get("complete"):
            return state
        atomic_json(state_path, state)
        accepted = directory / "accepted_rollouts.jsonl"
        dataset = directory / "sft_train.jsonl"
        checkpoints = directory / "checkpoint"
        start = time.monotonic()
        try:
            if not state.get("sampling_complete"):
                sampling_start = time.monotonic()
                self.ensure_model(pair["model"])
                command = self.adapter.sampling_command(
                    python=self.py,
                    config=self.execution_config(directory, pair["model"]),
                    data=self.train_data,
                    model=pair["model"]["id"],
                    base_url=self.base_url,
                    harness=pair["harness"],
                    attempts=int(self.config["rejection_sampling"]["attempts_per_item"]),
                    concurrency=int(self.config["rejection_sampling"]["concurrency"]),
                    quality_tier=self.config["rejection_sampling"]["quality_tier"],
                    max_turns=self.args.max_turns,
                    output_dir=directory,
                    limit=None,
                    resume=bool(getattr(self.args, "resume_parent_artifacts", False)),
                )
                self.services.run(
                    command, module=f"{training_id}.rejection_sampling", dry_run=False
                )
                state["sampling_complete"] = True
                self.cost("rejection_sampling", sampling_start, training_id=training_id)
                state["accepted_trajectory_ids"] = [r.get("sample_id") for r in _records(accepted)]
                atomic_json(state_path, state)
            if not state.get("dataset_complete"):
                count, source = self.services.prepare_training_dataset(
                    domain=self.spec.domain,
                    quality_tier=self.config["rejection_sampling"]["quality_tier"],
                    accepted_rollouts=accepted,
                    training_dataset=dataset,
                    train_data=self.train_data,
                    harness_name=pair["harness"],
                    sampling_config=self.config["rejection_sampling"],
                    training_config=self.training_config,
                    config=self.config,
                    val_data=self.val_data,
                    project_root=self.root,
                    validation_python=self.services.training_python,
                )
                if not count:
                    raise RuntimeError("no usable SFT records")
                state.update(
                    dataset_complete=True,
                    training_source=str(source),
                    dataset_hash=file_hash(dataset),
                )
                atomic_json(state_path, state)
            self.stop()
            if not state.get("checkpoint"):
                resumed = resumable_ms_swift_checkpoint(checkpoints)
                if resumed is not None:
                    trainer_state = json.loads((resumed / "trainer_state.json").read_text())
                    if float(trainer_state.get("epoch") or 0) + 1e-9 < float(
                        self.training_config.get("epochs", 1)
                    ):
                        resumed = None
                if resumed is None:
                    command = build_ms_swift_sft_command(
                        python=self.services.training_python,
                        data=dataset,
                        model=pair["model"]["path"],
                        output_dir=checkpoints,
                        training=self.training_config,
                    )
                    state["sft_started"] = True
                    atomic_json(state_path, state)
                    sft_start = time.monotonic()
                    self.services.run(command, module=f"{training_id}.sft", dry_run=False)
                    self.cost(
                        "sft",
                        sft_start,
                        training_id=training_id,
                        gpu_hours=(time.monotonic() - sft_start)
                        * int(self.training_config.get("num_processes", 1))
                        / 3600,
                        gpu_hours_basis="allocated_workers_times_wall",
                    )
                    resumed = resumable_ms_swift_checkpoint(checkpoints)
                    if resumed is None:
                        raise RuntimeError("SFT did not produce a complete checkpoint")
                state["checkpoint"] = str(resumed)
                atomic_json(state_path, state)
            post_model = describe_model(
                state["checkpoint"],
                project_root=self.root,
                cache_root=self.weights_root,
                common_base_id=pair["model"].get("common_base_id"),
                parent_ids=[pair["model"]["id"]],
                route="sft",
            )
            post_model["training_id"] = training_id
            state["output_model"] = post_model
            atomic_json(state_path, state)  # retain rejected or unevaluable checkpoint identities
            post = self.evaluate(pair, post_model)
            if post.get("evaluable", True):
                state["post"] = dict(post, phase="post", candidate_rank=rank)
            else:
                state["error"] = post["reason"]
        except (subprocess.CalledProcessError, OSError, ValueError, RuntimeError) as exc:
            state["error"] = str(exc)
            LOGGER.warning("training_id=%s failed; keeping pre-SFT pair: %s", training_id, exc)
        state["complete"] = True
        atomic_json(state_path, state)
        self.cost("training_path", start, training_id=training_id, candidate_rank=rank)
        return state

    def finalize(self, outcome, journal):
        winner = outcome["winner"]
        latest_archive = (
            json.loads(self.archive_path.read_text())
            if self.archive_path.exists()
            else copy.deepcopy(self.archive)
        )
        if (
            self.round_id in latest_archive["rounds"]
            and latest_archive.get("incumbent", {}).get("pair_id") != winner["id"]
        ):
            raise ValueError("cannot redeploy a superseded round over a newer incumbent")
        if self.round_id not in latest_archive["rounds"]:
            # Compare-and-swap the entire accepted pair; configuration is only a mirror.
            if latest_archive.get("incumbent") != self.snapshot.get("incumbent"):
                raise ValueError("another round changed incumbent before publication")
            models = [self.latest] + journal.data.get("models", [])
            models += [
                x["model"] for x in journal.data.get("merges", {}).values() if x.get("model")
            ]
            models += [
                x["output_model"]
                for x in journal.data["training"].values()
                if x.get("output_model")
            ]
            for model in models:
                latest_archive["models"][model["id"]] = model
            for batch in journal.data["batches"]:
                for result in batch.get("results", []):
                    proposal = result["proposal"]
                    latest_archive["harnesses"][proposal["candidate"]] = {
                        "id": proposal["candidate"],
                        "parent": self.parent,
                        "source_hash": proposal.get("harness_hash"),
                        "accepted": proposal["candidate"] == winner["harness"]
                        and outcome["status"] != "no_update",
                        "status": result["reason"],
                        "scores": result.get("pair", {}).get("scores"),
                        "model_ids": [
                            p["model"]["id"]
                            for p in journal.data["evaluations"].values()
                            if p["harness"] == proposal["candidate"]
                        ],
                        **proposal.get("lineage", {}),
                    }
            latest_archive["incumbent"] = {
                "harness": winner["harness"],
                "model": winner["model"],
                "pair_id": winner["id"],
                "served_model": winner["model"]["id"],
                "round_id": self.round_id,
            }
            latest_archive["rounds"][self.round_id] = {
                "status": outcome["status"],
                "reason": outcome["reason"],
                "journal": str(journal.path),
                "snapshot_id": self.snapshot["id"],
                "winner_pair_id": winner["id"],
            }
            atomic_json(self.archive_path, latest_archive)
        self.published = True
        set_active_harness(self.config_path, winner["harness"])
        publish_lineage_harnesses(
            self.registry_path,
            parent=self.parent,
            parent_path=self.parent_source,
            proposals=[p for b in journal.data["batches"] for p in b.get("proposals", [])],
            winner=winner["harness"],
            report_path=self.round_dir / "round_summary.json",
        )
        # Deploy the selected pre or post pair, including historical or merged weights.
        self.ensure_model(winner["model"])
        deployment = {
            "status": "verified_then_stopped" if self.args.stop_after_round else "running",
            "model": winner["model"]["id"],
            "model_path": winner["model"]["path"],
            "base_model": self.args.base_model_name,
            "base_url": self.base_url,
            "pid": self.server.process.pid,
            "pid_file": str(self.server.pid_path),
            "log": str(self.server.log_path),
        }
        self.account_serving()
        from src.diagnostics.mismatch import round_comparisons

        summary = {
            "round": self.args.round_number,
            "parent_harness": self.parent,
            "selected_harness": winner["harness"],
            "candidate_harness": outcome.get("selected", {}).get("harness"),
            "selection_decision": outcome["status"],
            "policy_updated": outcome["status"] != "no_update",
            # A whole round without an accepted child consumes patience even
            # when every proposal fails before rejection sampling/SFT.
            "policy_rollbacks": int(outcome["status"] == "no_update"),
            "harness_attempts": [
                p for b in journal.data["batches"] for p in b.get("proposals", [])
            ],
            "shortlisted_candidates": journal.data.get("shortlisted", []),
            "parent_validation": journal.data["incumbent_eval"],
            "policy_training": "completed" if journal.data["training"] else "skipped",
            "training_output": str(self.round_dir / "training"),
            "policy_validation": {
                "ranking": journal.data.get("ranking", []),
                "candidates": list(journal.data["training"].values()),
                "fallback": journal.data.get("fallback"),
                "outcome": outcome,
            },
            "policy_test": None,
            "mismatch": round_comparisons(journal.data, selected_pair_id=winner["id"]),
            "inheritance": self.inheritance_report(),
            "deployment": deployment,
            "archive": str(self.archive_path),
            "costs": ledger_summary(journal, self.round_dir, self.config),
        }
        atomic_json(self.round_dir / "deployment.json", deployment)
        atomic_json(self.round_dir / "round_summary.json", summary)
        if not self.args.stop_after_round:
            self.server.detach()
            self.server = None
            self.server_started = None
        return summary


def run_lineage(**kwargs):
    runtime = Runtime(**kwargs)
    if runtime.args.dry_run:
        meta = runtime.config["meta_evolver"]
        preview_stage = getattr(runtime.args, "stage", "evolve")
        preview = {
            "domain": runtime.adapter.name,
            "workflow": "LINEAGE",
            "stage": getattr(runtime.args, "stage", "evolve"),
            "proposal_batch_size": meta["proposal_batch_size"],
            "max_attempts_per_round": meta["max_attempts_per_round"],
            "validation": "full",
            "harness_shortlist": runtime.selection["max_shortlisted_children"],
            "third_harness": "within_non_policy_tolerances_of_second",
            "model_traversal": "all_available_compatible_frozen_models",
            "merge_weights": [0.5, 0.5],
            "max_training_paths": 0 if preview_stage == "model-selection" else 2,
            "second_training": None
            if preview_stage == "model-selection"
            else "close_pre_pairs_and_first_post_regresses",
            "parent_child_tolerances": {
                k: tolerance(runtime.selection, k) for k in quality_names(runtime.selection)
            },
            "pre_post_tolerance": runtime.selection["policy_validation_tolerance"],
        }
        if runtime.inheritance:
            preview.update(
                inheritance=runtime.inheritance,
                model_traversal="agent_requested_full_val"
                if runtime.inheritance["model"]["mode"] == "agent"
                else "all_compatible_singles_then_fixed_top_k_ties",
                merge_weights=None,
                harness_collection="agent"
                if runtime.inheritance["harness"]["mode"] == "agent"
                else runtime.inheritance["harness"]["strategy"],
            )
        if preview_stage == "model-selection":
            preview["selection_harnesses"] = runtime.args.selection_harness
            preview["generates_harnesses"] = False
            preview["publishes_incumbent"] = False
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return preview
    runtime.archive_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime.archive_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            runtime.initialize()
            if getattr(runtime.args, "stage", "evolve") == "model-selection":
                summary = runtime.select_models()
            else:
                summary = evolve_round(runtime, runtime.journal, runtime.config)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return summary
        finally:
            runtime.stop()
