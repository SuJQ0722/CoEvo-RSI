"""Behavioral coverage for collection, autonomous decisions, budgets and resume."""

import copy
import json
from pathlib import Path

import pytest

from src.meta_evolver.inheritance import collect_harnesses, inheritance_settings
from src.meta_evolver.lineage import Journal, select_model_pairs


def descriptor(name):
    return {
        "id": name,
        "effective_weights_hash": name,
        "common_base_id": "base",
        "architecture_hash": "arch",
        "tokenizer_hash": "tok",
        "layout_hash": "shape",
    }


class Backend:
    def __init__(self, mode, actions=(), harness_mode="agent", **settings):
        self.config = {
            "inheritance": {
                **settings,
                "harness": {"mode": harness_mode},
                "model": {"mode": mode, **settings.get("model", {})},
            }
        }
        self.base, self.latest = descriptor("base"), descriptor("latest")
        self.actions = list(actions)
        self.decisions, self.evaluations, self.merges = [], [], []
        self.cached = {}
        self.interrupt = None

    def models(self):
        return [self.latest, self.base, descriptor("a"), descriptor("b")]

    def evaluate(self, child, model):
        if self.interrupt == model["id"]:
            self.interrupt = None
            raise KeyboardInterrupt()
        key = child["harness"], model["id"]
        if key not in self.cached:
            self.evaluations.append(key)
            score = {"base": 0.6, "latest": 0.87, "a": 0.95, "b": 0.91, "merged": 0.97}[model["id"]]
            self.cached[key] = {
                "id": "_".join(key),
                "harness": child["harness"],
                "harness_hash": child["harness_hash"],
                "model": model,
                "scores": {"quality": score},
                "token_cost": 10,
                "artifact": {"rollouts": "/private/val/rollouts.jsonl"},
                "diagnostics": {
                    "mismatch": {
                        "rates": {"interface_mismatch_task_rate": {"rate": 0.1}},
                        "samples_path": "/private/val/mismatch_samples.jsonl",
                    }
                },
            }
        return self.cached[key]

    def admit(self, incumbent, pair):
        return pair.get("evaluable", True) and pair["scores"]["quality"] >= 0.8, "gate"

    def merge_candidates(self, parents, **options):
        self.merges.append(([m["id"] for m in parents], options))
        return descriptor("merged")

    def model_inheritance_action(self, child, step, inputs, prompt):
        self.decisions.append(copy.deepcopy(inputs))
        assert "/private/" not in json.dumps(inputs)
        assert "FULL val" in prompt
        assert "Actively use evaluation actions" in prompt
        assert "quality-first lexicographic ranking" in prompt
        assert "partial mismatch" in prompt
        return self.actions[step - 1]


CHILD = {"harness": "child", "harness_hash": "h_child"}
INCUMBENT = {"scores": {"quality": 0.8}}
SELECTION = {"quality_metrics": ["quality"]}


def run(backend, journal):
    return select_model_pairs(backend, journal, [CHILD], INCUMBENT, SELECTION)


@pytest.mark.parametrize("weighted", [False, True])
def test_model_agent_observes_subsets_and_uses_configured_ranking(tmp_path, weighted):
    class SplitBackend(Backend):
        def evaluate(self, child, model):
            result = super().evaluate(child, model)
            hand, algo = {
                "latest": ((40, 20), (60, 40)),
                "a": ((60, 30), (40, 30)),
                "b": ((40, 30), (80, 50)),
                "merged": ((55, 35), (60, 40)),
            }[model["id"]]
            result["scores"].update({
                "SourceA.agent_accuracy": hand[0],
                "SourceA.step_accuracy": hand[1],
                "SourceB.agent_accuracy": algo[0],
                "SourceB.step_accuracy": algo[1],
            })
            return result

    backend = SplitBackend("agent", actions=[
        {"action": "evaluate", "model_id": "a", "reason": "check SourceA strength"},
        {"action": "evaluate", "model_id": "b", "reason": "check SourceB strength"},
        {"action": "merge", "model_ids": ["a", "b"], "method": "linear",
         "weights": [0.7, 0.3], "reason": "evaluate complementary donors"},
        {"action": "select", "model_id": "b" if weighted else "merged",
         "reason": "best configured objective among evaluated models"},
    ])
    selection = dict(SELECTION, ranking_metric_groups=[
        ["SourceA.agent_accuracy", "SourceA.step_accuracy"],
        ["SourceB.agent_accuracy", "SourceB.step_accuracy"],
    ])
    if weighted:
        names = [name for group in selection["ranking_metric_groups"] for name in group]
        selection["ranking_metric_groups"] = [names]
        selection["ranking_metric_weights"] = {
            name: 7 if name.endswith("step_accuracy") else 3 for name in names
        }
    journal = Journal(tmp_path / "journal.json", {})
    ranked = select_model_pairs(backend, journal, [CHILD], INCUMBENT, selection)
    assert ranked[0]["model"]["id"] == ("b" if weighted else "merged")
    observed = {row["model_id"]: row for row in backend.decisions[2]["observations"]}
    assert observed["a"]["quality_score"] == ([36] if weighted else [45, 35])
    assert observed["b"]["quality_score"] == ([46] if weighted else [35, 65])
    assert observed["b"]["scores"]["SourceB.agent_accuracy"] == 80
    assert backend.merges == [(["a", "b"], {"method": "linear", "weights": [0.7, 0.3]})]


@pytest.mark.parametrize("harness_mode", ["agent", "predefined"])
def test_predefined_exact_top_k_and_resume(tmp_path, harness_mode):
    backend = Backend("predefined", harness_mode=harness_mode)
    journal = Journal(tmp_path / "journal.json", {})
    ranking = run(backend, journal)
    assert ranking[0]["model"]["id"] == "merged"
    assert backend.merges == [
        (["a", "b", "latest"], {"method": "ties", "density": 0.2, "scale": 1.0})
    ]
    assert len(backend.evaluations) == 5
    assert not backend.decisions
    assert run(backend, Journal(journal.path, {})) == ranking
    assert len(backend.merges) == 1


def test_fixed_k_does_not_shrink_with_insufficient_history(tmp_path):
    backend = Backend("predefined", model={"predefined": {"top_k": 4}})
    journal = Journal(tmp_path / "journal.json", {})
    assert run(backend, journal)[0]["model"]["id"] == "a"
    assert not backend.merges
    state = journal.data["model_inheritance"]["h_child"]
    assert state["merge_status"] == "insufficient_donors"
    assert state["available_donors"] == 3


@pytest.mark.parametrize("harness_mode", ["agent", "predefined"])
def test_agent_evaluates_merges_then_selects_from_real_feedback(tmp_path, harness_mode):
    backend = Backend(
        "agent",
        harness_mode=harness_mode,
        actions=[
            {"action": "evaluate", "model_id": "a", "reason": "Try old policy"},
            {
                "action": "merge",
                "model_ids": ["latest", "a"],
                "method": "ties",
                "density": 0.5,
                "scale": 0.8,
                "reason": "Resolve sign conflict",
            },
            {"action": "select", "model_id": "merged", "reason": "Validated improvement"},
        ],
    )
    journal = Journal(tmp_path / "journal.json", {})
    assert run(backend, journal)[0]["model"]["id"] == "merged"
    assert backend.evaluations == [("child", "latest"), ("child", "a"), ("child", "merged")]
    assert backend.decisions[1]["observations"][1]["scores"] == {"quality": 0.95}
    assert backend.decisions[2]["remaining"]["evaluations"] == 6
    assert journal.data["model_inheritance"]["h_child"]["selection_source"] == "agent"


def test_agent_must_select_best_observed_quality_without_mismatch_config(tmp_path):
    backend = Backend(
        "agent",
        actions=[
            {"action": "evaluate", "model_id": "a", "reason": "Compare"},
            {"action": "select", "model_id": "latest", "reason": "Prefer this training start"},
            {"action": "select", "model_id": "a", "reason": "Highest quality"},
        ],
    )
    assert run(backend, Journal(tmp_path / "j.json", {}))[0]["model"]["id"] == "a"
    assert "quality-first" in backend.decisions[2]["history"][1]["result"]["error"]


def test_agent_must_prefer_quality_over_lower_mismatch_within_tolerance(tmp_path):
    backend = Backend(
        "agent",
        actions=[
            {"action": "evaluate", "model_id": "a", "reason": "Compare"},
            {"action": "select", "model_id": "a", "reason": "Better interface match"},
            {"action": "select", "model_id": "latest", "reason": "Higher quality"},
        ],
    )
    original_evaluate = backend.evaluate

    def evaluate(child, model):
        result = original_evaluate(child, model)
        result["scores"]["quality"] = {"latest": 0.95, "a": 0.94}.get(
            model["id"], result["scores"]["quality"]
        )
        rate = {"latest": 0.20, "a": 0.19}.get(model["id"], 0.30)
        result["diagnostics"] = {
            "mismatch": {
                "metric_version": "model_harness_mismatch_v3",
                "rates": {"interface_mismatch_turn_rate": {"rate": rate}},
                "coverage": {"complete_interface_turns": {"rate": 1.0}},
                "identity": {"tasks": {"sha256": "same-val"}},
            }
        }
        return result

    backend.evaluate = evaluate
    selection = {
        "quality_metrics": ["quality"],
        "primary_quality_metric": "quality",
        "quality_tolerance": 0.02,
        "mismatch_metric": "interface_mismatch_turn_rate",
        "mismatch_coverage_metric": "complete_interface_turns",
        "epsilon": 1e-12,
    }
    ranking = select_model_pairs(
        backend,
        Journal(tmp_path / "j.json", {}),
        [CHILD],
        INCUMBENT,
        selection,
    )

    assert ranking[0]["model"]["id"] == "latest"
    assert "quality-first" in backend.decisions[2]["history"][1]["result"]["error"]


def test_agent_rejects_unmeasured_selection_and_test_split(tmp_path):
    backend = Backend(
        "agent",
        actions=[
            {"action": "select", "model_id": "a", "reason": "Unmeasured"},
            {"action": "evaluate", "model_id": "a", "split": "test", "reason": "Forbidden split"},
            {"action": "select", "model_id": "latest", "reason": "Use measured fallback"},
        ],
    )
    run(backend, Journal(tmp_path / "j.json", {}))
    assert "evaluated admissible" in backend.decisions[1]["history"][0]["result"]["error"]
    assert "unsupported" in backend.decisions[2]["history"][1]["result"]["error"]
    assert backend.evaluations == [("child", "latest")]


def test_budget_cap_falls_back_explicitly(tmp_path):
    backend = Backend(
        "agent",
        model={"agent": {"max_actions": 2, "max_evaluations": 1}},
        actions=[
            {"action": "evaluate", "model_id": "a", "reason": "Try old"},
            {"action": "evaluate", "model_id": "b", "reason": "Over budget"},
        ],
    )
    journal = Journal(tmp_path / "j.json", {})
    assert run(backend, journal)[0]["model"]["id"] == "a"
    state = journal.data["model_inheritance"]["h_child"]
    assert state["selection_source"] == "budget_fallback"
    assert "budget exhausted" in state["actions"][1]["result"]["error"]
    assert len(backend.evaluations) == 2


def test_interrupt_reuses_reserved_agent_decision(tmp_path):
    backend = Backend(
        "agent",
        actions=[
            {"action": "evaluate", "model_id": "a", "reason": "Try old"},
            {"action": "select", "model_id": "a", "reason": "Good start"},
        ],
    )
    backend.interrupt = "a"
    journal = Journal(tmp_path / "j.json", {})
    with pytest.raises(KeyboardInterrupt):
        run(backend, journal)
    assert len(backend.decisions) == 1
    assert run(backend, Journal(journal.path, {}))[0]["model"]["id"] == "a"
    assert len(backend.decisions) == 2


def test_predefined_harness_retrieval_is_order_independent():
    inputs = {
        "history": [
            {"id": "old", "source": {"fix.py": "sql parser syntax repair"}},
            {"id": "new", "source": {"fix.py": "unrelated room device lights"}},
        ],
        "validation_evidence": [{"outcome": {"error": "sql parser syntax"}}],
    }
    settings = {"strategy": "failure_similarity", "max_references": 1}
    chosen, receipt = collect_harnesses(inputs, settings)
    assert chosen[0]["id"] == "old"
    reverse = dict(inputs, history=list(reversed(inputs["history"])))
    assert collect_harnesses(reverse, settings) == (chosen, receipt)
    assert collect_harnesses(inputs, dict(settings, max_references=0))[0] == []


@pytest.mark.parametrize(
    "supplied",
    [
        {"mode": "typo"},
        {"harness": {"mode": "typo"}},
        {"model": {"mode": "typo"}},
        {"model": {"mode": None}},
        {"harness": {"mode": []}},
        {"model": {"predefined": {"top_k": 1}}},
        {"model": {"predefined": {"density": 0}}},
        {"harness": {"max_references": -1}},
        {"model": {"agent": {"max_actions": True}}},
        {"model": {"agent": {"merge_methods": ["unknown"]}}},
        {"typo": 1},
    ],
)
def test_invalid_inheritance_configuration_fails(supplied):
    with pytest.raises((ValueError, TypeError)):
        inheritance_settings({"inheritance": supplied})


@pytest.mark.parametrize("harness_mode", ["agent", "predefined"])
@pytest.mark.parametrize("model_mode", ["agent", "predefined"])
def test_curator_reference_selection_depends_only_on_harness_mode(
    tmp_path, monkeypatch, harness_mode, model_mode
):
    from src.meta_evolver import context

    response = tmp_path / "response.json"
    packet = {"target": {}, "references": [], "avoid": [], "constraints": {}}
    response.write_text(json.dumps(packet))
    monkeypatch.setattr(context, "run_meta_text", lambda **kwargs: response)
    args = {
        "config": {
            "inheritance": {"harness": {"mode": harness_mode}, "model": {"mode": model_mode}}
        },
        "project_root": tmp_path,
        "output_dir": tmp_path,
        "inputs": {"history": [{"id": "old"}], "incumbent": "latest_h", "archive_snapshot_id": "a"},
    }
    if harness_mode == "predefined":
        with pytest.raises(ValueError, match="exact collected"):
            context.curate_context(**args)
    else:
        assert context.curate_context(**args)["references"] == []
    packet["references"] = [{"harness_id": "old"}]
    response.write_text(json.dumps(packet))
    assert context.curate_context(**args)["harness_mode"] == harness_mode


def test_curator_normalizes_annotated_incumbent_before_reference_limit(
    tmp_path, monkeypatch
):
    from src.meta_evolver import context

    incumbent = "round_11_20260919T084655Z"
    historical_ids = [f"historical_{index}" for index in range(5)]
    response = tmp_path / "response.json"
    response.write_text(
        json.dumps(
            {
                "target": {},
                "references": [
                    {"harness_id": f"{incumbent} (current incumbent)"},
                    *[{"harness_id": harness_id} for harness_id in historical_ids],
                ],
                "avoid": [],
                "constraints": {},
            }
        )
    )
    monkeypatch.setattr(context, "run_meta_text", lambda **kwargs: response)

    packet = context.curate_context(
        config={
            "inheritance": {
                "harness": {"mode": "agent", "max_references": 5},
                "model": {"mode": "agent"},
            }
        },
        project_root=tmp_path,
        output_dir=tmp_path,
        inputs={
            "history": [{"id": harness_id} for harness_id in historical_ids],
            "incumbent": incumbent,
            "archive_snapshot_id": "snapshot",
        },
    )

    assert packet["references"][0]["harness_id"] == incumbent


@pytest.mark.parametrize(
    "legacy,expected", [("fully_agent", "agent"), ("full_predefined", "predefined")]
)
def test_legacy_shared_mode_and_component_override(legacy, expected):
    config = {"inheritance": {"mode": legacy}}
    settings = inheritance_settings(config)
    assert settings["harness"]["mode"] == settings["model"]["mode"] == expected
    config["inheritance"]["model"] = {"mode": "agent"}
    original = copy.deepcopy(config)
    settings = inheritance_settings(config)
    assert settings["harness"]["mode"] == expected
    assert settings["model"]["mode"] == "agent"
    assert "mode" not in settings
    assert config == original


def test_missing_section_keeps_legacy_flow_but_explicit_section_defaults_to_agent():
    assert inheritance_settings({}) is None
    settings = inheritance_settings({"inheritance": {}})
    assert settings["harness"]["mode"] == settings["model"]["mode"] == "agent"


@pytest.mark.parametrize(
    "env,override,expected",
    [
        ({}, {}, ("agent", "agent")),
        ({"HARNESS_INHERITANCE_MODE": "predefined"}, {}, ("agent", "agent")),
        ({"MODEL_INHERITANCE_MODE": "predefined"}, {}, ("agent", "agent")),
        ({"INHERITANCE_MODE": "full_predefined"}, {}, ("agent", "agent")),
        (
            {"INHERITANCE_MODE": "full_predefined", "HARNESS_INHERITANCE_MODE": "agent"},
            {},
            ("agent", "agent"),
        ),
        ({}, {"mode": "full_predefined"}, ("agent", "agent")),
        (
            {},
            {"mode": "full_predefined", "model": {"mode": "agent"}},
            ("agent", "agent"),
        ),
        (
            {"MODEL_INHERITANCE_MODE": "agent"},
            {"model": {"mode": "predefined"}},
            ("agent", "predefined"),
        ),
    ],
)
def test_composed_config_resolves_independent_modes(tmp_path, monkeypatch, env, override, expected):
    import yaml

    from src.config import load_config

    for name in ("INHERITANCE_MODE", "HARNESS_INHERITANCE_MODE", "MODEL_INHERITANCE_MODE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    base = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    (tmp_path / "base.yaml").write_text(base.read_text())
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump({"extends": "base.yaml", "inheritance": override}))
    settings = inheritance_settings(load_config(config_path))
    assert (settings["harness"]["mode"], settings["model"]["mode"]) == expected
