from pathlib import Path

import pytest

from src.config import load_config
from src.domains import EvaluationResult, get_domain_plugin
from src.domains.who_when.data import WhoWhenDataLoader

ROOT = Path(__file__).resolve().parents[1]


def test_loader_implementations_are_owned_by_domains():
    assert WhoWhenDataLoader.__module__ == "src.domains.who_when.data"


@pytest.mark.parametrize(
    "domain,kind", [("who_when", "failure_detection"), ("rbac", "rbac"), ("homebench", "homebench")]
)
def test_registry_resolves_domain_owned_components(domain, kind):
    plugin = get_domain_plugin(load_config(ROOT / f"configs/{domain}.yaml"))
    assert (plugin.name, plugin.harness_kind) == (domain, kind)


@pytest.mark.parametrize("domain", ["databench", "toolmaze"])
def test_archived_domains_are_not_registered(domain):
    with pytest.raises(ValueError, match="unsupported domain plugin"):
        get_domain_plugin(domain)


def test_declarative_selection_uses_homebench_metrics():
    config = load_config(ROOT / "configs/homebench.yaml")
    plugin = get_domain_plugin(config)
    rows = [{"sample_id": "a", "task_digest": "same-task"}]
    parent = EvaluationResult({"em": 0.4, "f1": 0.7}, rows)
    assert plugin.select(
        parent, EvaluationResult({"em": 0.5, "f1": 0.7}, rows), config["selection"]
    ).accepted
    assert not plugin.select(
        parent, EvaluationResult({"em": 0.5, "f1": 0.6}, rows), config["selection"]
    ).accepted


@pytest.mark.parametrize("domain", ["who_when", "rbac", "homebench"])
def test_domain_smoke_contracts_are_dispatched_by_plugin(domain):
    report = get_domain_plugin(domain).smoke_test(
        "initial_harness", package=f"harnesses.{domain}", root=ROOT / f"harnesses/{domain}"
    )
    assert report.passed, report.to_dict()
