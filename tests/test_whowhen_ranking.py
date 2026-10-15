"""Step-weighted validation ranking throughout model selection."""

import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml

from src.domains.round_adapter import EvaluationArtifact, get_round_adapter
from src.domains.selection import quality_score
from src.domains.who_when.evaluator import evaluate
from src.meta_evolver.lineage import rank_pairs
from src.meta_evolver.model_inheritance import _pair_observation


@pytest.fixture
def selection():
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / "configs/who_when.yaml").read_text())["selection"]


def pair(name, *, overall=(50, 30), cost=100):
    return {
        "id": name,
        "harness_hash": "same-harness",
        "model": {"id": name, "effective_weights_hash": name},
        "scores": {"agent_accuracy": overall[0], "step_accuracy": overall[1]},
        "quality_score": [-999],  # Old cached ordering must have no effect.
        "token_cost": cost,
    }


def test_ranking_uses_overall_validation_metrics_only(selection):
    names = {name for group in selection["ranking_metric_groups"] for name in group}
    assert names == {"agent_accuracy", "step_accuracy"}


@pytest.mark.parametrize("final", [False, True])
def test_step_weight_dominates(selection, final):
    high_step = pair("step", overall=(40, 40))
    high_agent = pair("agent", overall=(60, 30))
    assert quality_score(high_step["scores"], selection) == (40,)
    assert quality_score(high_agent["scores"], selection) == (39,)
    assert rank_pairs([high_agent, high_step], selection, final=final)[0] is high_step
    adapter = get_round_adapter("who_when")
    assert adapter.quality_score(high_step["scores"], selection) == (40,)


def test_equal_score_uses_cost(selection):
    first = pair("first")
    cheaper = pair("cheap", cost=1)
    assert rank_pairs([first, cheaper], selection)[0] is cheaper


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf")])
def test_invalid_weights_rejected(selection, weight):
    selection["ranking_metric_weights"]["step_accuracy"] = weight
    with pytest.raises(ValueError, match="finite and positive"):
        quality_score(pair("a")["scores"], selection)


def test_missing_or_nonfinite_metric_cannot_silently_fall_back(selection):
    scores = pair("a")["scores"]
    scores.pop("agent_accuracy")
    with pytest.raises(ValueError, match="ranking metrics are missing"):
        quality_score(scores, selection)
    scores["agent_accuracy"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        quality_score(scores, selection)


def test_official_scores_reach_merge_agent(tmp_path, selection):
    records = [
        {
            "sample_id": f"val:{i}",
            "source_dataset": source,
            "gold": {"responsible_agent": "a", "decisive_step": 1},
            "prediction": {"responsible_agent": "a", "evidence_step_ids": [step]},
            "error": error,
        }
        for i, (source, step, error) in enumerate([
            ("AgentErrorBench", 1, None),
            ("AgentErrorBench", 2, None),
            ("AgentErrorBench", 1, "max turns"),
            ("TraceElephant", 1, None),
        ])
    ]
    result = evaluate(records)
    # Failed records remain in the official denominator.
    assert result.metrics["agent_accuracy"] == pytest.approx(75)
    assert result.metrics["step_accuracy"] == pytest.approx(50)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(result.diagnostics))
    adapter = get_round_adapter("who_when")
    loaded = adapter.load_result(EvaluationArtifact(tmp_path, tmp_path / "unused", metrics_path))
    assert quality_score(loaded.metrics, selection) == pytest.approx((0.3 * 75 + 0.7 * 50,))
    candidate = pair("candidate", overall=(75, 50))
    candidate["artifact"] = {"metrics": str(metrics_path)}
    backend = SimpleNamespace(admit=lambda *_: (True, "within_tolerance"))
    observation = _pair_observation(backend, candidate, candidate, selection)
    assert observation["quality_score"] == pytest.approx([0.3 * 75 + 0.7 * 50])
    assert "artifact" not in observation
    assert rank_pairs([pair("other"), candidate], selection)[0] is candidate
