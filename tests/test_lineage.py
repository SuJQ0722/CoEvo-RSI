"""Behavioral acceptance tests for the shared LINEAGE state machine."""

import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.meta_evolver.lineage import (
    Journal,
    evolve_round,
    fallback_decision,
    model_pair_admission,
    rank_pairs,
    shortlist_harnesses,
)
from src.meta_evolver.weights import describe_model, harness_hash, merge_half, merge_reason


def config(width=3, budget=9):
    return {
        "meta_evolver": {
            "proposal_batch_size": width,
            "max_attempts_per_round": budget,
            "generation_concurrency": width,
        },
        "selection": {
            "quality_metrics": ["safe_ex", "ac_f1"],
            "safe_ex_tolerance": 0.01,
            "ac_f1_tolerance": 0.01,
            "policy_validation_tolerance": 0.005,
            "epsilon": 1e-12,
            "max_shortlisted_children": 2,
        },
    }


def test_journal_resume_allows_config_changes(tmp_path):
    path = tmp_path / "journal.json"
    frozen = {
        "config": {
            "meta_evolver": {
                "backend": "codex",
                "codex": {"model": "old-meta", "reasoning_effort": "high"},
            },
            "rollout": {"temperature": 0.0},
        }
    }
    Journal(path, frozen)

    changed_model = copy.deepcopy(frozen)
    changed_model["config"]["meta_evolver"]["codex"]["model"] = "new-meta"
    Journal(path, changed_model)

    changed_rollout = copy.deepcopy(changed_model)
    changed_rollout["config"]["rollout"]["temperature"] = 0.7
    resumed = Journal(path, changed_rollout)
    assert resumed.data["identity"] == frozen


def test_journal_resume_can_disable_selected_identity_checks(tmp_path):
    path = tmp_path / "journal.json"
    frozen = {
        "splits": {"train": "old-data"},
        "dependencies": {"train": "old-environment"},
        "model_hash": "old-model",
        "harness_hash": "old-harness",
        "code_hash": "fixed-code",
    }
    Journal(path, frozen)
    changed = {
        **frozen,
        "splits": {"train": "new-data"},
        "dependencies": {"train": "new-environment"},
        "model_hash": "new-model",
        "harness_hash": "new-harness",
    }
    resumed = Journal(
        path,
        changed,
        ignored_identity_fields={"splits", "dependencies", "model_hash", "harness_hash"},
    )
    assert resumed.data["identity"] == frozen


def pair(name, score=0.8, *, ac=None, cost=10, model="latest", harness=None, phase="pre", rank=1):
    return {
        "id": name,
        "harness": harness or name,
        "harness_hash": harness or name,
        "model": {"id": model, "effective_weights_hash": model},
        "scores": {"safe_ex": score, "ac_f1": score if ac is None else ac},
        "token_cost": cost,
        "phase": phase,
        "candidate_rank": rank,
    }


class FakeBackend:
    round_id = "r1"
    latest = {"id": "latest", "effective_weights_hash": "latest"}  # noqa: RUF012 -- immutable fixture descriptor

    def __init__(self, success_batch=1, width=3, drop=0.006, second_score=0.9, no_post=False):
        self.success_batch, self.width, self.drop = success_batch, width, drop
        self.second_score, self.no_post = second_score, no_post
        self.events, self.finished = [], set()
        self.calls = {"curate": 0, "briefs": 0, "edit": 0, "summary": 0, "train": 0}
        self.eval_cache = {}
        self.interrupt_train = False

    def incumbent_eval(self):
        return pair("incumbent", 0.8)

    def curate(self):
        self.calls["curate"] += 1
        return {"id": "initial"}

    def briefs(self, context, feedback, batch):
        self.calls["briefs"] += 1
        self.events.append(("briefs", len(feedback), list(batch["slots"])))
        assert len(feedback) == int(batch["id"][-2:]) - 1
        return {"initial": context, "feedback": copy.deepcopy(feedback)}, [
            {"id": str(slot)} for slot in batch["slots"]
        ]

    def edit(self, context, brief, slot):
        self.calls["edit"] += 1
        time.sleep(0.001)
        self.finished.add(slot)
        return {
            "id": str(slot),
            "harness": str(slot),
            "harness_hash": str(slot),
            "slot": slot,
            "valid": True,
        }

    def evaluate(self, child, model):
        key = (child["harness"], model["id"])
        if key not in self.eval_cache:
            slot = int(child["harness"])
            batch = (slot - 1) // self.width + 1
            assert set(range((batch - 1) * self.width + 1, batch * self.width + 1)).issubset(
                self.finished
            )
            score = 0.9 if batch == self.success_batch else 0.7
            if model["id"] == "old":
                score = 0.92 if slot % self.width == 1 else self.second_score
            if model["id"] == "mix":
                score = 0.91
            self.eval_cache[key] = pair(
                f"{slot}_{model['id']}", score, model=model["id"], harness=str(slot)
            )
        return self.eval_cache[key]

    def admit(self, incumbent, candidate):
        return candidate["scores"]["safe_ex"] > incumbent["scores"]["safe_ex"], "domain_gate"

    accept = admit

    def summarize(self, batch):
        self.calls["summary"] += 1
        assert not any(item["accepted"] for item in batch["results"])
        return {"id": batch["id"], "text": "failure attribution"}

    def models(self):
        return [self.latest, {"id": "old", "effective_weights_hash": "old"}]

    def merge(self, best, latest):
        return {"id": "mix", "effective_weights_hash": "mix"}

    def train(self, candidate):
        if self.interrupt_train:
            self.interrupt_train = False
            raise KeyboardInterrupt()
        self.calls["train"] += 1
        self.events.append(("train", candidate["candidate_rank"], candidate["model"]["id"]))
        if self.no_post:
            return {"post": None}
        score = (
            candidate["scores"]["safe_ex"] - self.drop if candidate["candidate_rank"] == 1 else 0.95
        )
        return {
            "post": pair(
                candidate["id"] + "_post",
                score,
                model=candidate["model"]["id"] + "post",
                harness=candidate["harness"],
                rank=candidate["candidate_rank"],
                phase="post",
            )
        }

    def finalize(self, outcome, journal):
        self.events.append(("publish", outcome["status"]))
        return outcome


@pytest.mark.parametrize(
    "width,budget,success,edit_count,summary_count",
    [
        (8, 8, 1, 8, 0),
        (8, 8, 99, 8, 0),
        (3, 9, 1, 3, 0),
        (3, 9, 2, 6, 1),
        (3, 9, 3, 9, 2),
        (3, 9, 99, 9, 2),
    ],
)
def test_one_scheduler_lazy_briefs_and_batch_boundaries(
    tmp_path, width, budget, success, edit_count, summary_count
):
    backend = FakeBackend(success, width)
    journal = Journal(tmp_path / "journal.json", {})
    result = evolve_round(backend, journal, config(width, budget))
    assert backend.calls["edit"] == edit_count
    assert backend.calls["summary"] == summary_count
    assert backend.calls["curate"] == 1
    assert backend.calls["briefs"] == (edit_count // width)
    assert backend.events[-1][0] == "publish"
    if success == 99:
        assert result["status"] == "no_update"
        assert not backend.calls["train"]
    else:
        # Historical and merged models are evaluated only on the successful top2 harnesses.
        traversed = {h for h, m in backend.eval_cache if m != "latest"}
        assert traversed == {p["harness"] for p in journal.data["shortlisted"]}
        assert len(traversed) == 2


@pytest.mark.parametrize(
    "third_score,third_ac,expected",
    [(0.89, 0.89, 3), (0.889, 0.9, 2), (0.9, 0.889, 2), (0.9, 0.9, 3)],
)
def test_third_harness_uses_each_non_policy_tolerance(third_score, third_ac, expected):
    selection = config()["selection"]
    selection.update(max_shortlisted_children=3, policy_validation_tolerance=0.5)
    candidates = [
        pair("first", 0.99),
        pair("second", 0.9),
        pair("third", third_score, ac=third_ac),
        pair("fourth", 0.88),
    ]
    result = shortlist_harnesses(candidates, selection)
    assert [p["harness"] for p in result] == ["first", "second", "third"][:expected]
    # The distance from first to second, and the policy threshold, do not gate third.
    selection["policy_validation_tolerance"] = 0
    assert shortlist_harnesses(candidates, selection) == result


@pytest.mark.parametrize("limit,count", [(1, 4), (2, 4), (3, 0), (3, 1), (3, 2)])
def test_shortlist_respects_smaller_limits_and_available_candidates(limit, count):
    selection = dict(config()["selection"], max_shortlisted_children=limit)
    candidates = [pair(str(i), 0.9) for i in range(count)]
    assert len(shortlist_harnesses(candidates, selection)) == min(limit, count)


@pytest.mark.parametrize("third_score,expected", [(0.891, 3), (0.889, 2)])
def test_round_searches_models_for_third_harness_only_when_close(tmp_path, third_score, expected):
    backend = FakeBackend(width=4)
    evaluate = backend.evaluate

    def scored(child, model):
        result = evaluate(child, model)
        if model["id"] == "latest":
            score = {"1": 0.95, "2": 0.9, "3": third_score, "4": 0.85}[child["harness"]]
            result["scores"] = {"safe_ex": score, "ac_f1": score}
        return result

    backend.evaluate = scored
    cfg = config(width=4, budget=4)
    cfg["selection"]["max_shortlisted_children"] = 3
    journal = Journal(tmp_path / "journal.json", {})
    evolve_round(backend, journal, cfg)
    traversed = {h for h, m in backend.eval_cache if m != "latest"}
    assert traversed == {str(i) for i in range(1, expected + 1)}
    # Three harness searches still feed the shared top-two model/harness training paths.
    assert len(journal.data["ranking"]) == 2


@pytest.mark.parametrize("domain,expected", [("who_when", 5.0), ("homebench", 0.01)])
def test_configured_non_policy_tolerances_use_domain_units(domain, expected):
    from src.config import load_config
    from src.meta_evolver.lineage import tolerance

    selection = load_config(Path(__file__).resolve().parents[1] / "configs" / f"{domain}.yaml")[
        "selection"
    ]
    assert selection["max_shortlisted_children"] == 3
    for name in selection["quality_metrics"]:
        assert tolerance(selection, name) == expected


def test_duplicates_and_invalid_children_consume_slots(tmp_path):
    backend = FakeBackend()

    def edit(context, brief, slot):
        backend.calls["edit"] += 1
        return {
            "harness": str(slot),
            "valid": slot % 3 == 0,
            "harness_hash": "incumbent",
            "reason": "bad_contract",
        }

    backend.edit = edit
    journal = Journal(tmp_path / "journal.json", {})
    result = evolve_round(backend, journal, config())
    assert result["status"] == "no_update"
    assert backend.calls["edit"] == 9
    assert not backend.eval_cache


@pytest.mark.parametrize(
    "gap,drop,post,trigger",
    [
        (0.01, 0.005, True, False),
        (0.01, 0.0051, True, True),
        (0.0101, 0.006, True, False),
        (0, 0.006, False, False),
        (0, 0.002, True, False),
        (0, -0.01, True, False),
    ],
)
def test_fallback_boundaries_use_own_pre_scores(gap, drop, post, trigger):
    first = pair("first", 0.9)
    second = pair("second", 0.9 - gap)
    trained = pair("post", 0.9 - drop, phase="post") if post else None
    assert fallback_decision(first, second, trained, config()["selection"])["triggered"] is trigger


def test_secondary_metric_also_controls_closeness_and_regression():
    selection = config()["selection"]
    first = pair("a", 0.9, ac=0.8)
    assert not fallback_decision(
        first, pair("b", 0.895, ac=0.82), pair("post", 0.89, ac=0.8), selection
    )["triggered"]
    assert fallback_decision(
        first, pair("b", 0.895, ac=0.8), pair("post", 0.901, ac=0.79), selection
    )["triggered"]


def test_training_compares_four_pairs_and_can_train_same_harness_twice(tmp_path):
    backend = FakeBackend()
    journal = Journal(tmp_path / "journal.json", {})
    result = evolve_round(backend, journal, config())
    assert backend.calls["train"] == 2
    assert result["winner"]["scores"]["safe_ex"] == 0.95
    assert result["status"] == "trained_pair"
    ranking = journal.data["ranking"]
    assert ranking[0]["model"]["id"] == "old"
    assert ranking[1]["model"]["id"] == "mix"
    assert ranking[0]["harness"] == ranking[1]["harness"]
    assert [event[2] for event in backend.events if event[0] == "train"] == ["old", "mix"]


def test_single_training_candidate_disables_fallback_without_narrowing_search(tmp_path):
    cfg = config()
    cfg["selection"]["max_training_candidates"] = 1
    backend = FakeBackend()
    journal = Journal(tmp_path / "journal.json", {})
    result = evolve_round(backend, journal, cfg)
    assert len(journal.data["ranking"]) == 2
    assert len(journal.data["shortlisted"]) == 2
    assert backend.calls["train"] == 1
    assert list(journal.data["training"]) == ["1"]
    decision = journal.data["fallback"]
    assert decision["close"] and decision["regressed"]
    assert not decision["triggered"]
    assert decision["reason"] == "training_candidate_limit"
    assert result["status"] == "untrained_pair"
    assert result["winner"]["candidate_rank"] == 1
    # Resume must neither resample candidate 1 nor start candidate 2.
    assert evolve_round(backend, journal, cfg) == result
    assert backend.calls["train"] == 1


@pytest.mark.parametrize("limit", [0, 3])
def test_invalid_training_candidate_limit_fails_before_work(tmp_path, limit):
    cfg = config()
    cfg["selection"]["max_training_candidates"] = limit
    backend = FakeBackend()
    with pytest.raises(ValueError, match="max_training_candidates"):
        evolve_round(backend, Journal(tmp_path / "journal.json", {}), cfg)
    assert not backend.events


@pytest.mark.parametrize("drop,no_post", [(0.002, False), (0.006, True)])
def test_small_drop_or_no_checkpoint_retains_pre_without_fallback(tmp_path, drop, no_post):
    backend = FakeBackend(drop=drop, no_post=no_post)
    journal = Journal(tmp_path / "journal.json", {})
    result = evolve_round(backend, journal, config())
    assert backend.calls["train"] == 1
    assert result["status"] == "untrained_pair"
    assert result["winner"]["scores"]["safe_ex"] == 0.92


def test_tie_order_quality_tokens_pre_then_rank():
    s = config()["selection"]
    pre = pair("pre", 0.8, phase="pre", rank=2)
    post = pair("post", 0.8, phase="post", rank=1)
    assert rank_pairs([post, pre], s, final=True)[0] == pre
    cheaper = pair("cheap", 0.8, cost=9, phase="post")
    assert rank_pairs([pre, cheaper], s, final=True)[0] == cheaper
    slight = pair("slight", 0.8001, cost=10000, phase="post")
    assert rank_pairs([pre, slight], s, final=True)[0] == slight
    first = pair("first", 0.8, rank=1)
    assert rank_pairs([pre, first], s, final=True)[0] == first
    unknown = pair("unknown", 0.8, cost=None)
    assert rank_pairs([pre, unknown], s, final=True)[0] == pre


def _with_turn_imr(candidate, rate, *, coverage=1.0, task_set="same-val"):
    candidate["diagnostics"] = {
        "mismatch": {
            "metric_version": "model_harness_mismatch_v3",
            "rates": {"interface_mismatch_turn_rate": {"rate": rate}},
            "coverage": {"complete_interface_turns": {"rate": coverage}},
            "identity": {"tasks": {"sha256": task_set}},
        }
    }
    return candidate


def test_model_ranking_prefers_quality_even_within_primary_tolerance():
    selection = dict(
        config()["selection"],
        primary_quality_metric="safe_ex",
        mismatch_metric="interface_mismatch_turn_rate",
        mismatch_coverage_metric="complete_interface_turns",
    )
    quality_leader = _with_turn_imr(pair("quality", 0.90), 0.20)
    match_leader = _with_turn_imr(pair("match", 0.89), 0.19)

    assert (
        rank_pairs([quality_leader, match_leader], selection, prefer_mismatch=True)[0]
        is quality_leader
    )


def test_model_ranking_protects_primary_metric_beyond_tolerance():
    selection = dict(
        config()["selection"],
        primary_quality_metric="safe_ex",
        mismatch_metric="interface_mismatch_turn_rate",
        mismatch_coverage_metric="complete_interface_turns",
    )
    quality_leader = _with_turn_imr(pair("quality", 0.90), 0.20)
    match_leader = _with_turn_imr(pair("match", 0.889), 0.0)

    assert (
        rank_pairs([quality_leader, match_leader], selection, prefer_mismatch=True)[0]
        is quality_leader
    )


def test_model_ranking_does_not_treat_partial_mismatch_as_comparable():
    selection = dict(
        config()["selection"],
        primary_quality_metric="safe_ex",
        mismatch_metric="interface_mismatch_turn_rate",
        mismatch_coverage_metric="complete_interface_turns",
    )
    quality_leader = _with_turn_imr(pair("quality", 0.90), 0.20)
    partial = _with_turn_imr(pair("partial", 0.89), 0.0, coverage=0.9)

    assert (
        rank_pairs([quality_leader, partial], selection, prefer_mismatch=True)[0] is quality_leader
    )


def test_model_admission_does_not_override_domain_gate_for_better_match():
    selection = dict(
        config()["selection"],
        primary_quality_metric="safe_ex",
        mismatch_metric="interface_mismatch_turn_rate",
        mismatch_coverage_metric="complete_interface_turns",
    )
    backend = FakeBackend()
    incumbent = _with_turn_imr(pair("incumbent", 0.80), 0.20)
    candidate = _with_turn_imr(pair("candidate", 0.795), 0.19)

    accepted, reason = model_pair_admission(backend, incumbent, candidate, selection)

    assert not accepted
    assert reason == "domain_gate"


def test_secondary_quality_precedes_mismatch_and_cost():
    selection = dict(config()["selection"], mismatch_metric="interface_mismatch_turn_rate")
    quality = _with_turn_imr(pair("quality", 0.9, ac=0.901, cost=100), 0.2)
    match = _with_turn_imr(pair("match", 0.9, ac=0.9, cost=1), 0.0)
    assert rank_pairs([match, quality], selection, prefer_mismatch=True)[0] is quality


@pytest.mark.parametrize(
    "coverage,task_set,expected",
    [
        (1.0, "same-val", "match"),
        (0.9, "same-val", "cheap"),
        (1.0, "other-val", "cheap"),
    ],
)
def test_mismatch_breaks_quality_ties_only_when_comparable(coverage, task_set, expected):
    selection = dict(config()["selection"], mismatch_metric="interface_mismatch_turn_rate")
    cheap = _with_turn_imr(pair("cheap", 0.9, cost=1), 0.2)
    match = _with_turn_imr(pair("match", 0.9, cost=100), 0.0, coverage=coverage, task_set=task_set)
    assert rank_pairs([cheap, match], selection, prefer_mismatch=True)[0]["id"] == expected


def test_resume_preserves_ranking_briefs_and_completed_training(tmp_path):
    backend = FakeBackend()
    path = tmp_path / "journal.json"
    journal = Journal(path, {"snapshot": "fixed"})
    backend.interrupt_train = True
    with pytest.raises(KeyboardInterrupt):
        evolve_round(backend, journal, config())
    frozen = copy.deepcopy(journal.data["ranking"])
    assert backend.calls["edit"] == 3
    evolve_round(backend, Journal(path, {"snapshot": "fixed"}), config())
    before = copy.deepcopy(backend.calls)
    evolve_round(backend, Journal(path, {"snapshot": "fixed"}), config())
    assert backend.calls == before
    assert json.loads(path.read_text())["ranking"] == frozen
    assert before == {"curate": 1, "briefs": 1, "edit": 3, "summary": 0, "train": 2}
    with pytest.raises(ValueError, match="inputs changed"):
        Journal(path, {"snapshot": "changed"})


def test_explicit_resume_allows_audited_controller_code_repair_only(tmp_path):
    path = tmp_path / "journal.json"
    frozen = {"snapshot": "fixed", "code_hash": "before"}
    Journal(path, frozen)

    resumed = Journal(
        path,
        {"snapshot": "fixed", "code_hash": "after"},
        allow_code_change=True,
    )
    assert resumed.data["identity"] == frozen
    assert resumed.data["resume_code_hashes"] == ["after"]

    with pytest.raises(ValueError, match="inputs changed"):
        Journal(
            path,
            {"snapshot": "different", "code_hash": "after"},
            allow_code_change=True,
        )


def test_resume_logs_code_and_environment_differences(tmp_path, caplog):
    path = tmp_path / "journal.json"
    frozen = {
        "code_hash": "before",
        "environment_version": {
            "domain_python": "/old/python",
            "packages": {"removed": "1", "same": "1", "updated": "1"},
        },
    }
    current = {
        "code_hash": "after",
        "environment_version": {
            "domain_python": "/new/python",
            "packages": {"added": "2", "same": "1", "updated": "2"},
        },
    }
    Journal(path, frozen)

    with (
        caplog.at_level("ERROR", logger="pipeline.lineage"),
        pytest.raises(ValueError, match="code_hash, environment_version"),
    ):
        Journal(path, current, allow_code_change=True)

    log = caplog.text
    assert "field=code_hash frozen=before current=after" in log
    assert "domain_python frozen=/old/python current=/new/python" in log
    assert 'packages added={"added": "2"}' in log
    assert 'packages removed={"removed": "1"}' in log
    assert 'package versions changed={"updated": {"current": "2", "frozen": "1"}}' in log


def test_resume_ignores_python_executable_alias_when_packages_match(tmp_path):
    path = tmp_path / "journal.json"
    frozen = {
        "code_hash": "before",
        "environment_version": {
            "domain_python": "/usr/bin/python",
            "packages": {"runtime": "1"},
        },
    }
    Journal(path, frozen)

    resumed = Journal(
        path,
        {
            "code_hash": "after",
            "environment_version": {
                "domain_python": "/usr/bin/python3",
                "packages": {"runtime": "1"},
            },
        },
        allow_code_change=True,
    )

    assert resumed.data["identity"] == frozen
    assert resumed.data["resume_code_hashes"] == ["after"]


def _tiny_model(path, values, dtype=None, tokenizer="same"):
    import torch
    from safetensors.torch import save_file

    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "tiny", "hidden_size": 2}))
    (path / "tokenizer.json").write_text(json.dumps({"vocab": tokenizer}))
    save_file(
        {"weight": torch.tensor(values, dtype=dtype or torch.float32)}, path / "model.safetensors"
    )
    return path


def test_fp32_effective_merge_and_common_base_constraints(tmp_path):
    import torch
    from safetensors.torch import load_file

    a = _tiny_model(tmp_path / "a", [[1, 3], [5, 7]], torch.bfloat16)
    b = _tiny_model(tmp_path / "b", [[3, 5], [7, 9]], torch.bfloat16)
    opts = {"project_root": tmp_path, "cache_root": tmp_path / "cache", "common_base_id": "base"}
    left, right = describe_model(a, **opts), describe_model(b, **opts)
    mixed = merge_half(left, right, project_root=tmp_path, cache_root=tmp_path / "cache")
    shard = next(Path(mixed["path"]).glob("model-*.safetensors"))
    tensor = load_file(shard)["weight"]
    assert torch.equal(tensor, torch.tensor([[2, 4], [6, 8]], dtype=torch.bfloat16))
    assert mixed["parent_weights"] == [0.5, 0.5]
    assert merge_reason(left, left) == "identical_effective_weights"
    assert (
        merge_reason(left, dict(right, common_base_id="other")) == "unproven_common_initialization"
    )
    assert (
        merge_reason(left, dict(right, tokenizer_hash="other"))
        == "incompatible_architecture_tokenizer_or_layout"
    )
    # Identical effective values deduplicate regardless of container dtype/path.
    c = _tiny_model(tmp_path / "c", [[1, 3], [5, 7]], torch.float32)
    assert describe_model(c, **opts)["effective_weights_hash"] == left["effective_weights_hash"]


def test_harness_dedup_ignores_reports_but_includes_runtime_assets(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for directory in (a, b):
        directory.mkdir()
        (directory / "__init__.py").write_text("def run():\n    return 1\n")
    (a / "evolution_report.md").write_text("report one")
    (b / "evolution_report.md").write_text("report two")
    (b / "lineage.json").write_text('{"reference_ids": []}')
    assert harness_hash(a) == harness_hash(b)
    (b / "config.yaml").write_text("overrides:\n  harness:\n    max_turns: 12\n")
    assert harness_hash(a) != harness_hash(b)


def test_meta_read_isolation_denies_siblings_absolute_paths_and_symlinks(tmp_path):
    from src.meta_evolver.isolation import isolated_command

    workspace = tmp_path / "editor"
    workspace.mkdir()
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    secret = sibling / "validation.json"
    secret.write_text("private")
    (workspace / "context.json").write_text("shared")
    (workspace / "escape").symlink_to(secret)
    script = """from pathlib import Path
assert Path('context.json').read_text() == 'shared'
for path in [Path(%r), Path('escape')]:
    try:
        path.read_text()
    except PermissionError:
        pass
    else:
        raise AssertionError('read boundary escaped')
print('isolated')
""" % str(secret)  # noqa: UP031 -- embedded Python template
    result = subprocess.run(
        isolated_command([sys.executable, "-c", script], workspace, tmp_path / "runtime"),
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )
    if "requires Landlock ABI" in result.stderr:
        pytest.skip("host kernel does not expose Landlock")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "isolated"


def test_model_identity_requires_dense_checkpoint_weights(tmp_path):
    model = tmp_path / "incomplete_checkpoint"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "tiny", "hidden_size": 2}')
    (model / "tokenizer.json").write_text('{"vocab": "same"}')
    with pytest.raises(ValueError, match="no dense model weights"):
        describe_model(model, project_root=tmp_path, cache_root=tmp_path / "cache")


@pytest.mark.parametrize("prefer_mismatch", [False, True])
def test_who_when_step_priority_ignores_legacy_mean_vectors(prefer_mismatch):
    from src.domains.round_adapter import get_round_adapter

    # Old configs listed Agent first but already declared Step as primary.
    selection = {
        "quality_metrics": ["agent_accuracy", "step_accuracy"],
        "primary_quality_metric": "step_accuracy",
        "quality_tolerance": 5,
        "mismatch_metric": "interface_mismatch_turn_rate",
        "epsilon": 1e-12,
    }
    adapter = get_round_adapter("who_when")
    # R4: C1 post had higher mean and lower IMR; C2 post had better Step.
    a = _with_turn_imr(pair("c1_post", cost=9336739, phase="post"), 0.0)
    b = _with_turn_imr(pair("c2_post", cost=13316157, phase="post"), 0.00381679389)
    a["scores"] = {"agent_accuracy": 50.3875968992, "step_accuracy": 25.1937984496}
    b["scores"] = {"agent_accuracy": 48.0620155039, "step_accuracy": 26.3565891473}
    for candidate in (a, b):
        agent, step = candidate["scores"].values()
        candidate["quality_score"] = [(agent + step) / 2, agent, step]
    assert rank_pairs([a, b], selection, prefer_mismatch=prefer_mismatch)[0] is b
    pre = _with_turn_imr(pair("c1_pre"), 0.0)
    pre["scores"] = {"agent_accuracy": 48.8372093023, "step_accuracy": 26.3565891473}
    assert rank_pairs([a, b, pre], selection, final=True, prefer_mismatch=prefer_mismatch)[0] is pre
    new_selection = dict(selection, quality_metrics=["step_accuracy", "agent_accuracy"])
    assert adapter.quality_score(b["scores"], new_selection) == (26.3565891473, 48.0620155039)


def test_standalone_model_selection_reuses_evolution_traversal_without_training(tmp_path):
    from src.meta_evolver.lineage import select_model_pairs

    backend = FakeBackend()
    backend.finished = {1, 2, 3}
    journal = Journal(tmp_path / "journal.json", {})
    ranking = select_model_pairs(
        backend, journal, [pair("1")], backend.incumbent_eval(), config()["selection"]
    )
    assert {model for _, model in backend.eval_cache} == {"latest", "old", "mix"}
    assert ranking[0]["model"]["id"] == "old"
    assert not any(backend.calls.values())
    assert not backend.events
    assert "outcome" not in journal.data and not journal.data["training"]
    backend.evaluate = lambda *args: pytest.fail("completed ranking should be reused")
    assert (
        select_model_pairs(
            backend, journal, [pair("1")], backend.incumbent_eval(), config()["selection"]
        )
        == ranking
    )
