import json

from src.domains.round_adapter import get_round_adapter


def test_rbac_adapter_loads_six_category_evaluation_artifact(tmp_path):
    adapter = get_round_adapter("rbac")
    artifact = adapter.artifact(tmp_path, "parent_val")
    artifact.root.mkdir(parents=True)
    artifact.rollouts.write_text('{"sample_id": "rbac:1"}\n', encoding="utf-8")
    metrics = {
        "safe_ex": 0.75,
        "ac_f1": 0.8,
        "safe_deny": 0.9,
        "violation_rate": 0.05,
        "over_refusal_rate": 0.1,
        "llm_rate": 1.0,
        "rule_rate": 0.0,
        "rule_accuracy": 0.0,
    }
    artifact.metrics.write_text(
        json.dumps(
            {
                "protocol": "rbac_text2sql_six_category_v2",
                "harness": "initial_harness",
                "total": 1,
                **metrics,
            }
        ),
        encoding="utf-8",
    )

    assert adapter.load_result(artifact).metrics == metrics
    assert adapter.artifact_complete(artifact, expected=1)
