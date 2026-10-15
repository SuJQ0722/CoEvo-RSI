from src.domains.who_when.runtime.evaluation import (
    StrictAttributionJudge,
    build_judge,
    evaluate,
    format_score_summary,
)


def test_evaluation_module_cli_does_not_import_itself_before_runpy():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-m",
            "src.domains.who_when.runtime.evaluation",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--predictions" in result.stdout
    assert "found in sys.modules" not in result.stderr


def test_who_when_package_keeps_plugin_exports():
    from src.domains.registry import get_domain_plugin
    from src.domains.who_when import PLUGIN, WhoWhenPlugin

    assert isinstance(PLUGIN, WhoWhenPlugin)
    assert get_domain_plugin("who_when") is PLUGIN


def test_score_summary_places_step_accuracy_directly_below_agent_accuracy():
    metrics = evaluate([])
    assert format_score_summary(metrics).splitlines() == [
        "Agent Accuracy: 0.00%",
        "Step Accuracy:  0.00%",
    ]


def test_unparseable_predictions_reduce_coverage():
    metrics = evaluate(
        [
            {
                "gold": {"responsible_agent": "Agent", "decisive_step": 1},
                "prediction": {"responsible_agent": "Agent", "evidence_step_ids": [1]},
            },
            {
                "gold": {"responsible_agent": "Agent", "decisive_step": 2},
                "prediction": None,
            },
            {
                "gold": {"responsible_agent": "Agent", "decisive_step": 3},
                "prediction": {"responsible_agent": "", "evidence_step_ids": []},
            },
        ],
    )
    assert metrics.coverage == 1 / 3
    assert metrics.agent_accuracy == 1 / 3 * 100.0
    assert metrics.step_accuracy == 1 / 3 * 100.0


def test_who_and_when_official_metrics_use_all_records_as_denominator():
    metrics = evaluate(
        [
            {
                "source_dataset": "Who&When",
                "sample_id": "who_and_when:Algorithm-Generated:1",
                "gold": {
                    "is_failure": True,
                    "decisive_step": 3,
                    "responsible_agent": "WebSurfer",
                },
                "prediction": {
                    "is_failure": True,
                    "evidence_step_ids": [3, 4],
                    "responsible_agent": "WebSurfer",
                },
            },
            {
                "source_dataset": "Who&When",
                "sample_id": "who_and_when:Hand-Crafted:2",
                "gold": {
                    "is_failure": True,
                    "decisive_step": 2,
                    "responsible_agent": "Assistant",
                },
                "prediction": None,
            },
        ]
    )
    assert metrics.protocol == "who_and_when_official"
    assert metrics.agent_accuracy == 50.0
    assert metrics.step_accuracy == 50.0
    assert metrics.coverage == 0.5
    assert metrics.by_split["Algorithm-Generated"]["agent_accuracy"] == 100.0
    assert metrics.by_split["Hand-Crafted"]["agent_accuracy"] == 0.0


def test_operational_failures_are_reported_without_changing_scores():
    metrics = evaluate(
        [
            {
                "gold": {"responsible_agent": "Agent", "decisive_step": 1},
                "prediction": None,
                "error": "max turns reached without final_answer",
                "turns": 10,
                "inspected_step_ids": [1, 2],
            },
            {
                "gold": {"responsible_agent": "Agent", "decisive_step": 2},
                "prediction": {"responsible_agent": "Agent", "evidence_step_ids": [2]},
                "error": "",
                "turns": 2,
                "inspected_step_ids": [2],
                "token_usage_available": True,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        ]
    )
    assert metrics.step_accuracy == 50.0
    assert metrics.completed_total == 1
    assert metrics.agent_accuracy_completed == 100.0
    assert metrics.step_accuracy_completed == 100.0
    assert metrics.error_count == 1
    assert metrics.max_turn_failures == 1
    assert metrics.avg_turns == 6.0
    assert metrics.avg_inspected_steps == 1.5
    assert metrics.avg_total_tokens == 120.0


def test_step_only_records_do_not_reduce_agent_accuracy():
    metrics = evaluate(
        [
            {
                "gold": {"responsible_agent": "", "decisive_step": 7},
                "prediction": {"responsible_agent": "", "evidence_step_ids": [7]},
            },
            {
                "gold": {"responsible_agent": "Planner", "decisive_step": 3},
                "prediction": {
                    "responsible_agent": "Planner",
                    "evidence_step_ids": [3],
                },
            },
        ]
    )
    assert metrics.agent_labeled_total == 1
    assert metrics.agent_accuracy == 100.0
    assert metrics.step_accuracy == 100.0


def test_build_judge_uses_offline_final_answer_matching():
    assert isinstance(build_judge({}), StrictAttributionJudge)
