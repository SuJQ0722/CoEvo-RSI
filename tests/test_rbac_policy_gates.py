from pathlib import Path

import pytest

from src.config import load_config
from src.domains.base import EvaluationResult
from src.domains.rbac.selection import select
from src.meta_evolver.lineage import fallback_decision, rank_pairs
from src.meta_evolver.evolution import (
    load_harness_tree,
    record_selection,
    write_harness_tree,
)


ROOT = Path(__file__).resolve().parents[1]


def _scores(*, safe_ex: float, ac_f1: float, llm_rate: float = 0.8) -> dict:
    return {
        "safe_ex": safe_ex,
        "ac_f1": ac_f1,
        "llm_rate": llm_rate,
        "rule_rate": 0.2,
        "rule_accuracy": 0.9,
    }


def _pair(pair_id: str, scores: dict, *, phase: str) -> dict:
    return {
        "id": pair_id,
        "harness": "child",
        "harness_hash": "child-hash",
        "model": {"effective_weights_hash": f"{pair_id}-weights"},
        "scores": scores,
        "token_cost": 1,
        "phase": phase,
        "candidate_rank": 1,
    }


@pytest.mark.parametrize("protected_metric", ["safe_ex", "ac_f1"])
def test_pre_sft_harness_gate_allows_only_one_point_regression(protected_metric):
    config = load_config(ROOT / "configs/rbac.yaml")["selection"]
    parent_scores = _scores(safe_ex=0.9, ac_f1=0.9)
    within_tolerance = _scores(safe_ex=0.9, ac_f1=0.9, llm_rate=0.7)
    within_tolerance[protected_metric] -= 0.01
    outside_tolerance = dict(within_tolerance)
    outside_tolerance[protected_metric] -= 0.0001

    accepted = select(
        EvaluationResult(parent_scores),
        EvaluationResult(within_tolerance),
        config,
    )
    rejected = select(
        EvaluationResult(parent_scores),
        EvaluationResult(outside_tolerance),
        config,
    )

    assert accepted.accepted
    assert not rejected.accepted
    assert "tolerance" in rejected.reason


@pytest.mark.parametrize("training_limit", [1, 2])
def test_post_sft_gate_uses_remote_tolerance_against_its_pre_sft_pair(training_limit):
    selection = load_config(ROOT / "configs/rbac.yaml")["selection"]
    selection["max_training_candidates"] = training_limit
    pre_sft = _pair("pre", _scores(safe_ex=0.72, ac_f1=0.72), phase="pre")
    post_sft = _pair("post", _scores(safe_ex=0.7149, ac_f1=0.7149), phase="post")
    second = _pair("second", _scores(safe_ex=0.719, ac_f1=0.719), phase="pre")

    decision = fallback_decision(pre_sft, second, post_sft, selection)

    assert selection["policy_validation_tolerance"] == 0.005
    assert decision["regressed"]
    assert decision["triggered"] is (training_limit == 2)
    if training_limit == 1:
        assert decision["reason"] == "training_candidate_limit"


def test_regressed_post_sft_pair_keeps_the_pre_sft_pair_eligible():
    selection = load_config(ROOT / "configs/rbac.yaml")["selection"]
    pre_sft = _pair("pre", _scores(safe_ex=0.72, ac_f1=0.72), phase="pre")
    post_sft = _pair("post", _scores(safe_ex=0.71, ac_f1=0.71), phase="post")

    assert rank_pairs([pre_sft, post_sft], selection, final=True)[0] is pre_sft


def test_improved_post_sft_checkpoint_takes_priority_over_pre_sft_pair():
    selection = load_config(ROOT / "configs/rbac.yaml")["selection"]
    pre_sft = _pair("pre", _scores(safe_ex=0.72, ac_f1=0.72), phase="pre")
    post_sft = _pair("post", _scores(safe_ex=0.73, ac_f1=0.73), phase="post")

    assert rank_pairs([pre_sft, post_sft], selection, final=True)[0] is post_sft


def test_harness_only_promotion_updates_registry_active_node(tmp_path):
    registry = tmp_path / "tree.json"
    tree = load_harness_tree(registry)
    tree["nodes"]["child"] = {
        "parent": "initial_harness",
        "generation": 1,
        "status": "pareto_candidate",
        "path": "harnesses/rbac/candidates/child",
        "smoke": None,
        "selections": [],
    }
    write_harness_tree(registry, tree)
    report = tmp_path / "policy_validation_harness_fallback.json"
    report.write_text("{}\n", encoding="utf-8")

    record_selection(
        registry,
        "initial_harness",
        "child",
        "promote_pre_sft_harness",
        report,
    )

    updated = load_harness_tree(registry)
    assert updated["active"] == "child"
    assert updated["nodes"]["child"]["status"] == "active"
    assert updated["nodes"]["initial_harness"]["status"] == "archived"
