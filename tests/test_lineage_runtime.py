"""CPU-only integration of durable runtime ports; no serving or paid MetaAgent calls."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from safetensors.torch import save_file

from src.domains.base import EvaluationResult
from src.domains.round_adapter import EvaluationArtifact
from src.meta_evolver.round_runtime import Runtime, validate_splits
from src.meta_evolver.weights import describe_model, harness_hash
from src.utils.persistence import atomic_json


class Adapter:
    name = "unit_domain"
    prefilter_requires_improvement = True
    supports_lineage = True

    def quality_score(self, scores, selection):
        return (scores["score"],)

    def artifact(self, root, name):
        root = root / name
        return EvaluationArtifact(root, root / "samples.jsonl", root / "metrics.json")

    def artifact_complete(self, artifact, expected):
        if not artifact.metrics.exists() or not artifact.rollouts.exists():
            return False
        rows = [json.loads(line) for line in artifact.rollouts.read_text().splitlines()]
        return len(rows) == expected and len({r["sample_id"] for r in rows}) == expected

    def load_result(self, artifact):
        return EvaluationResult(json.loads(artifact.metrics.read_text()))

    def sampling_command(self, **kwargs):
        return [
            "sample",
            kwargs["model"],
            kwargs["harness"],
            str(kwargs["output_dir"]),
            str(kwargs["config"]),
        ]


def tiny_model(path, values):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type":"tiny"}')
    (path / "tokenizer.json").write_text('{"vocab":"same"}')
    save_file({"weight": torch.tensor(values, dtype=torch.float32)}, path / "model.safetensors")
    return path


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from src.meta_evolver import round_runtime as module

    cfg = {
        "project": {"domain": "unit_domain"},
        "harness": {
            "kind": "unit_domain",
            "package": "harnesses.unit_domain",
            "root": str(tmp_path / "harnesses"),
            "base": "initial_harness",
            "active": "initial_harness",
        },
        "meta_evolver": {
            "registry_path": str(tmp_path / "registry.json"),
            "proposal_batch_size": 3,
            "max_attempts_per_round": 9,
            "generation_concurrency": 3,
        },
        "selection": {
            "quality_metrics": ["score"],
            "max_shortlisted_children": 2,
            "quality_tolerance": 0.01,
            "policy_validation_tolerance": 0.005,
        },
        "model": {"base": str(tiny_model(tmp_path / "base", [1.0, 2.0]))},
        "rollout": {"concurrency": 2, "temperature": 0.3},
        "rejection_sampling": {
            "attempts_per_item": 3,
            "concurrency": 2,
            "quality_tier": "strict",
            "accepted_per_item": 2,
        },
        "training": {"epochs": 1, "num_processes": 1},
    }
    source = tmp_path / "harnesses" / "initial_harness"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("def run():\n    return 1\n")
    child = tmp_path / "harnesses" / "candidates" / "child"
    child.mkdir(parents=True)
    (child / "__init__.py").write_text("def run():\n    return 2\n")
    splits = {}
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps({"sample_id": f"{split}{i}"}) + "\n" for i in range(4)))
        splits[split] = path
    cfg["data"] = {"splits": {k: str(v) for k, v in splits.items()}}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    events = []
    adapter = Adapter()
    monkeypatch.setattr(module, "get_round_adapter", lambda cfg: adapter)
    monkeypatch.setattr(
        module,
        "get_domain_plugin",
        lambda cfg: SimpleNamespace(
            select=lambda parent, child, selection: SimpleNamespace(
                accepted=child.metrics["score"] > parent.metrics["score"], reason="quality"
            )
        ),
    )

    def evaluate(adapter, **kwargs):
        assert kwargs["limit"] is None
        assert kwargs["split"] in {"train", "val"}
        runtime_cfg = yaml.safe_load(kwargs["config"].read_text())
        assert runtime_cfg["rollout"]["temperature"] == 0.3
        assert runtime_cfg["model"]["revision"] != "unknown"
        events.append(("eval", kwargs["harness"], kwargs["model"], kwargs["data"]))
        artifact = kwargs["artifact"]
        artifact.root.mkdir(parents=True, exist_ok=True)
        rows = [
            dict(json.loads(line), total_tokens=10, token_usage_available=True)
            for line in kwargs["data"].read_text().splitlines()
        ]
        artifact.rollouts.write_text("".join(json.dumps(r) + "\n" for r in rows))
        atomic_json(artifact.metrics, {"score": 0.9})
        return adapter.load_result(artifact)

    def run(command, **kwargs):
        events.append(tuple(command[:4]))
        if command[0] == "sample":
            directory = Path(command[3])
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "accepted_rollouts.jsonl").write_text('{"sample_id":"train0"}\n')
            effective = yaml.safe_load(Path(command[4]).read_text())
            assert effective["rejection_sampling"]["accepted_per_item"] == 2
        if command[0] == "sft":
            checkpoint = tiny_model(Path(command[2]) / "checkpoint-1", [2.0, 3.0])
            atomic_json(checkpoint / "trainer_state.json", {"global_step": 1, "epoch": 1})

    def prepare(**kwargs):
        kwargs["training_dataset"].write_text('{"messages": []}\n')
        return 1, kwargs["accepted_rollouts"]

    monkeypatch.setattr(
        module,
        "build_ms_swift_sft_command",
        lambda **kw: ["sft", kw["model"], str(kw["output_dir"])],
    )
    args = SimpleNamespace(
        model="base",
        round_number=1,
        max_turns=4,
        backend="codex",
        training_limit=None,
        stop_after_round=True,
        base_model_name="base",
        dry_run=False,
    )
    services = SimpleNamespace(
        project_path=lambda p: Path(p),
        evaluate_with_adapter=evaluate,
        run=run,
        prepare_training_dataset=prepare,
        training_python="python",
    )
    rt = Runtime(
        args=args,
        config=cfg,
        config_path=cfg_path,
        project_root=tmp_path,
        output_root=tmp_path / "output",
        round_dir=tmp_path / "output" / "rounds" / "round1",
        domain_python="python",
        model_source=cfg["model"]["base"],
        base_url="http://localhost:8000/v1",
        vllm_settings={},
        training_config=cfg["training"],
        services=services,
        train_data=splits["train"],
        val_data=splits["val"],
        test_data=splits["test"],
    )
    rt.initialize()
    monkeypatch.setattr(rt, "ensure_model", lambda model: events.append(("load", model["id"])))
    rt.events = events
    return rt


def test_cache_uses_full_val_and_real_weight_and_decoding_identity(runtime):
    rt = runtime
    child = {"harness": "child", "harness_hash": harness_hash(rt.spec.path_for("child"))}
    a = rt.evaluate(child, rt.latest)
    b = rt.evaluate(child, rt.latest)
    assert a["eval_id"] == b["eval_id"]
    assert a["token_cost"] == 40
    diagnostics = a["diagnostics"]["mismatch"]
    assert diagnostics["sample_count"] == 4
    assert diagnostics["identity"]["pair_id"] == a["id"]
    assert diagnostics["identity"]["split"] == "val"
    assert diagnostics["rates"]["interface_mismatch_task_rate"]["rate"] is None
    assert Path(diagnostics["report_path"]).exists()
    assert Path(diagnostics["samples_path"]).exists()
    assert b["diagnostics"] == a["diagnostics"]
    assert set(a["artifact"]) == {"root", "rollouts", "metrics"}
    assert a["scores"] == {"score": 0.9}
    assert len([e for e in rt.events if e[0] == "eval"]) == 1
    donor = describe_model(
        tiny_model(rt.root / "donor", [4.0, 5.0]),
        project_root=rt.root,
        cache_root=rt.weights_root,
        common_base_id=rt.latest["common_base_id"],
    )
    c = rt.evaluate(child, donor)
    assert c["eval_id"] != a["eval_id"]
    assert len([e for e in rt.events if e[0] == "eval"]) == 2
    manifest = json.loads((Path(a["artifact"]["root"]).parent / "identity.json").read_text())
    assert manifest["seed"] == 0
    assert len(manifest["tasks"]["task_ids"]) == 4
    rt.val_data.write_text(rt.val_data.read_text().replace("val0", "changed"))
    with pytest.raises(ValueError, match="dataset changed"):
        rt.evaluate(child, rt.latest)


@pytest.mark.parametrize("harness_mode", ["agent", "predefined"])
@pytest.mark.parametrize("model_mode", ["agent", "predefined"])
def test_curator_reuses_incumbent_validation_evidence(
    runtime, monkeypatch, harness_mode, model_mode
):
    from src.meta_evolver import round_runtime as module

    rt = runtime
    rt.config["inheritance"] = {
        "harness": {"mode": harness_mode, "strategy": "recent", "max_references": 1},
        "model": {"mode": model_mode},
    }
    rt.inheritance = module.inheritance_settings(rt.config)
    rt.snapshot["history_packets"] = [
        {"id": "old", "generation": 1},
        {"id": "new", "generation": 2},
    ]
    incumbent = rt.incumbent_eval()
    rt.journal.data["incumbent_eval"] = incumbent
    rt.journal.save()
    rt.plugin.build_evidence = lambda source, output: output.write_text(source.read_text())
    rt.plugin.meta_user_prompt = lambda config: "unit domain"
    captured = {}

    def curate_context(**kwargs):
        captured.update(kwargs["inputs"])
        return {"id": "context"}

    monkeypatch.setattr(module, "curate_context", curate_context)

    assert rt.curate() == {"id": "context"}
    assert [card["id"] for card in captured["history"]] == (
        ["new"] if harness_mode == "predefined" else ["old", "new"]
    )
    assert "validation_evidence" in captured
    assert "train_evidence" not in captured
    assert {row["sample_id"] for row in captured["validation_evidence"]} == {
        "val0",
        "val1",
        "val2",
        "val3",
    }
    evaluations = [event for event in rt.events if event[0] == "eval"]
    assert len(evaluations) == 1
    assert evaluations[0][3] == rt.val_data


def test_training_uses_own_pre_model_and_resumes_completed_artifacts(runtime):
    rt = runtime
    child = {"harness": "child", "harness_hash": harness_hash(rt.spec.path_for("child"))}
    donor = describe_model(
        tiny_model(rt.root / "donor", [4.0, 5.0]),
        project_root=rt.root,
        cache_root=rt.weights_root,
        common_base_id=rt.latest["common_base_id"],
    )
    pre = dict(rt.evaluate(child, donor), candidate_rank=2)
    result = rt.train(pre)
    assert result["post"] is not None
    assert result["post"]["candidate_rank"] == 2
    assert result["input_model_id"] == donor["id"]
    sampled = [e for e in rt.events if e[0] == "sample"]
    trained = [e for e in rt.events if e[0] == "sft"]
    assert sampled[0][1:3] == (donor["id"], "child")
    assert trained[0][1] == donor["path"]
    rt.train(pre)
    assert len([e for e in rt.events if e[0] == "sample"]) == 1
    assert len([e for e in rt.events if e[0] == "sft"]) == 1
    assert all(e[3] != rt.test_data for e in rt.events if e[0] == "eval")


def test_no_checkpoint_preserves_pre_and_records_failed_path(runtime, monkeypatch):
    rt = runtime
    pair = dict(rt.evaluate({"harness": "child"}, rt.latest), candidate_rank=1)
    monkeypatch.setattr(
        rt.services, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sampling failed"))
    )
    state = rt.train(pair)
    assert state["post"] is None
    assert state["complete"]
    assert "sampling failed" in state["error"]
    assert rt.train(pair) == state


def test_publication_atomically_commits_pair_and_reconciles_config(runtime, monkeypatch):
    rt = runtime
    pre = rt.evaluate({"harness": "child"}, rt.latest)
    inc = rt.evaluate({"harness": rt.parent}, rt.latest)
    rt.journal.data["incumbent_eval"] = inc
    outcome = {"status": "untrained_pair", "reason": "quality", "selected": pre, "winner": pre}
    rt.journal.data["outcome"] = outcome
    rt.journal.data["batches"] = [
        {
            "proposals": [
                {
                    "candidate": "child",
                    "harness_hash": pre["harness_hash"],
                    "valid": True,
                    "source_path": str(rt.spec.path_for("child")),
                }
            ],
            "results": [
                {
                    "proposal": {
                        "candidate": "child",
                        "harness_hash": pre["harness_hash"],
                        "valid": True,
                    },
                    "pair": pre,
                    "reason": "quality",
                }
            ],
        }
    ]
    rt.journal.save()

    def fail_after_commit(model):
        raise RuntimeError("deployment interrupted")

    monkeypatch.setattr(rt, "ensure_model", fail_after_commit)
    with pytest.raises(RuntimeError, match="deployment interrupted"):
        rt.finalize(outcome, rt.journal)
    archive = json.loads(rt.archive_path.read_text())
    assert archive["incumbent"]["harness"] == "child"
    assert archive["incumbent"]["model"]["id"] == pre["model"]["id"]
    assert archive["rounds"][rt.round_id]["status"] == "untrained_pair"
    assert yaml.safe_load(rt.config_path.read_text())["harness"]["active"] == "child"
    # Reconciliation is idempotent, even after the public active mirror changed.
    with pytest.raises(RuntimeError, match="deployment interrupted"):
        rt.finalize(outcome, rt.journal)
    assert json.loads(rt.archive_path.read_text()) == archive


def test_split_overlap_is_rejected_before_execution(tmp_path):
    paths = [tmp_path / f"{i}.jsonl" for i in range(3)]
    for path in paths:
        path.write_text('{"sample_id":"overlap"}\n')
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(*paths)


def test_val_test_overlap_is_allowed_without_allowing_training_overlap(tmp_path):
    train, val, test = [tmp_path / f"{name}.jsonl" for name in ("train", "val", "test")]
    train.write_text('{"sample_id":"train-only"}\n')
    val.write_text('{"sample_id":"evaluation"}\n')
    test.write_text('{"sample_id":"evaluation"}\n')

    identities = validate_splits(train, val, test)
    assert identities["val"]["task_ids"] == identities["test"]["task_ids"]

    identities = validate_splits(train, test, test)
    assert identities["val"] == identities["test"]

    train.write_text('{"sample_id":"evaluation"}\n')
    with pytest.raises(ValueError, match="train/val"):
        validate_splits(train, val, test)


def test_split_identity_accepts_portable_id_field(tmp_path):
    train, val, test = [tmp_path / f"{name}.jsonl" for name in ("train", "val", "test")]
    train.write_text('{"id":"portable-train"}\n')
    val.write_text('{"id":"portable-eval"}\n')
    test.write_text('{"id":"portable-eval"}\n')

    identities = validate_splits(train, val, test)

    assert identities["train"]["task_ids"] == ["portable-train"]
    assert identities["val"]["task_ids"] == ["portable-eval"]


def test_mismatch_failure_does_not_change_validation_result(runtime, monkeypatch):
    from src.diagnostics import mismatch

    def broken_report(*args, **kwargs):
        raise ValueError("invalid diagnostic contract")

    monkeypatch.setattr(mismatch, "write_report", broken_report)
    pair = runtime.evaluate({"harness": "child"}, runtime.latest)
    assert pair["evaluable"]
    assert pair["scores"] == {"score": 0.9}
    assert pair["diagnostics"]["mismatch"]["status"] == "unavailable"
    assert len([event for event in runtime.events if event[0] == "eval"]) == 1


def test_standalone_selection_writes_diagnostics_without_training_or_publication(
    runtime, monkeypatch
):
    from src.domains.base import SmokeReport

    rt = runtime
    rt.args.selection_harness = ["child"]
    rt.plugin.smoke_test = lambda name, **kwargs: SmokeReport(name, True, ())
    original = rt.config_path.read_bytes()
    monkeypatch.setattr(rt, "train", lambda *args: pytest.fail("standalone selection cannot train"))
    monkeypatch.setattr(
        rt, "finalize", lambda *args: pytest.fail("standalone selection cannot publish")
    )
    report = rt.select_models()
    assert report["stage"] == "model-selection" and not report["published"]
    assert report["pair_pool"]
    assert report["mismatch"]["pairs"]
    assert not rt.journal.data["training"] and not rt.journal.data["batches"]
    assert rt.config_path.read_bytes() == original
    assert not rt.archive_path.exists()
    assert json.loads((rt.round_dir / "model_selection.json").read_text()) == report
    eval_count = sum(event[0] == "eval" for event in rt.events)
    rt.select_models()
    assert sum(event[0] == "eval" for event in rt.events) == eval_count


@pytest.mark.parametrize("harness_mode", ["agent", "predefined"])
@pytest.mark.parametrize("model_mode", ["agent", "predefined"])
def test_runtime_dispatches_independent_model_mode(runtime, monkeypatch, harness_mode, model_mode):
    from src.domains.base import SmokeReport
    from src.meta_evolver.inheritance import inheritance_settings

    rt = runtime
    rt.config["inheritance"] = {"harness": {"mode": harness_mode}, "model": {"mode": model_mode}}
    rt.inheritance = inheritance_settings(rt.config)
    rt.args.selection_harness = ["child"]
    rt.plugin.smoke_test = lambda name, **kwargs: SmokeReport(name, True, ())
    monkeypatch.setattr(rt, "admit", lambda *args: (True, "fixture gate"))
    monkeypatch.setattr(rt, "train", lambda *args: pytest.fail("selection must not train"))
    actions = []

    def choose(child, step, inputs, prompt):
        actions.append(inputs)
        return {"action": "select", "model_id": rt.latest["id"], "reason": "Use tested base"}

    monkeypatch.setattr(rt, "model_inheritance_action", choose)
    report = rt.select_models()
    assert report["ranking"][0]["model"]["id"] == rt.latest["id"]
    assert report["inheritance"]["settings"]["harness"]["mode"] == harness_mode
    assert report["inheritance"]["settings"]["model"]["mode"] == model_mode
    assert len(actions) == int(model_mode == "agent")
    search = report["inheritance"]["searches"][0]
    assert search["mode"] == model_mode
    if model_mode == "predefined":
        assert search["merge_status"] == "insufficient_donors"
    else:
        assert search["selection_source"] == "agent"
    assert not rt.journal.data["training"]


def test_runtime_ties_port_records_and_reuses_merge(runtime, tmp_path):
    from src.meta_evolver.weights import TensorFiles

    rt = runtime
    parents = [
        describe_model(
            tiny_model(tmp_path / name, values),
            project_root=tmp_path,
            cache_root=rt.weights_root,
            common_base_id=rt.base["effective_weights_hash"],
        )
        for name, values in (("donor_a", [3.0, 4.0]), ("donor_b", [2.0, 6.0]))
    ]
    mixed = rt.merge_candidates(parents, method="ties", density=1, scale=1)
    torch.testing.assert_close(
        TensorFiles(Path(mixed["path"])).tensor("weight"), torch.tensor([2.5, 5.0])
    )
    assert len(rt.journal.data["merges"]) == 1
    assert rt.merge_candidates(parents, method="ties", density=1, scale=1) == mixed
    assert sum(entry["stage"] == "merge" for entry in rt.journal.data["costs"]) == 1
