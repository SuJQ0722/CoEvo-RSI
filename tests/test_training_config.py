from pathlib import Path

from src.config import (
    harness_spec,
    load_config,
    write_resolved_config,
)


def test_training_config_uses_ms_swift_full_parameter_sft():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")
    who_when = load_config(Path(__file__).resolve().parents[1] / "configs/who_when.yaml")
    training = config["training"]
    assert training["backend"] == "ms_swift"
    assert training["submodule_path"] == "ms-swift"
    assert training["method"] == "ms_swift_sft"
    assert training["tuner_type"] == "full"
    assert training["epochs"] == 1
    assert training["learning_rate"] == 5e-6
    assert training["batch_size"] == 1
    assert training["gradient_accumulation_steps"] == 4
    assert training["max_length"] == 65536
    assert training["gradient_checkpointing"] is True
    assert training["optimizer"] == "adamw_torch"
    assert training["lr_scheduler_type"] == "cosine"
    assert training["warmup_ratio"] == 0.03
    assert training["num_processes"] == 4
    assert training["sequence_parallel_size"] == 2
    assert (
        training["batch_size"]
        * (training["num_processes"] // training["sequence_parallel_size"])
        * training["gradient_accumulation_steps"]
        == 8
    )
    assert training["padding_free"] is False
    assert training["deepspeed"] == "zero3"
    assert training["truncation_strategy"] == "delete"
    assert training["agent_template"] == "qwen3_5"
    assert who_when["selection"]["quality_metrics"] == [
        "agent_accuracy",
        "step_accuracy",
    ]
    assert who_when["selection"]["quality_tolerance"] == 5
    assert who_when["selection"]["policy_validation_tolerance"] == 1
    assert who_when["selection"]["promotion_split"] == "validation"
    assert who_when["selection"]["max_shortlisted_children"] == 3
    assert who_when["selection"]["ranking_metric_groups"] == [["agent_accuracy", "step_accuracy"]]
    assert who_when["selection"]["mismatch_metric"] == "interface_mismatch_turn_rate"
    sampling = who_when["rejection_sampling"]
    assert sampling["quality_tier"] == "strict"
    assert sampling["repair_strategy"] == "none"
    assert sampling["balance_by_val_difficulty"] is True
    assert sampling["balance_reference_data"] == "data/whowhen/val.jsonl"
    assert sampling["balance_max_growth_fraction"] == 0.5
    assert sampling["balance_max_repeats_per_record"] == 3
    assert sampling["balance_axis_weights"] == {
        "trace_length": 1.0,
        "decisive_position": 2.0,
        "agent_count": 1.0,
    }
    assert config["evolution"]["max_rounds"] == 10
    assert config["evolution"]["policy_rollback_patience"] == 3
    assert who_when["data"]["splits"] == {
        "train": "data/whowhen/train.jsonl",
        "val": "data/whowhen/val.jsonl",
        "test": "data/whowhen/test_whowhen.jsonl",
        "test_traceelephant": "data/whowhen/test_traceelephant.jsonl",
        "test_whowhen_pro": "data/whowhen/test_whowhen_pro.jsonl",
    }
    assert who_when["paths"] == {"output": "output/who_when"}
    assert who_when["evolution"]["max_rounds"] == 20
    assert who_when["resume_validation"] == {
        "data": False,
        "model": False,
        "harness": False,
    }


def test_rbac_training_uses_four_workers_without_offload_and_short_context():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/rbac.yaml")
    training = config["training"]
    assert training["num_processes"] == 4
    assert training["sequence_parallel_size"] == 1
    assert training["deepspeed"] == "zero3"
    assert training["max_length"] == 8192
    assert config["rejection_sampling"]["quality_tier"] == "balanced"
    assert config["rejection_sampling"]["repair_strategy"] == "verified_only"
    assert config["rejection_sampling"]["require_sql_execution"] is True
    assert config["rejection_sampling"]["balance_strategy"] == "downsample"
    assert config["execution"]["environment_prefix"] == ".venvs/rbac"


def test_homebench_uses_thinking_policy():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/homebench.yaml")
    assert config["data"]["splits"] == {
        "train": "data/homebench/train.jsonl",
        "val": "data/homebench/val.jsonl",
        "test": "data/homebench/test.jsonl",
    }
    assert config["data"]["manifest"] == "data/homebench/manifest.json"
    assert config["data"]["expected_counts"] == {
        "train": 10000,
        "val": 2000,
        "test": 4414,
    }
    assert config["vllm"]["tensor_parallel_size"] == 1
    assert config["vllm"]["data_parallel_size"] == 4
    assert config["training"]["num_processes"] == 4
    assert config["training"]["sequence_parallel_size"] == 2
    assert config["rollout"]["enable_thinking"] is True
    assert config["harness"]["max_turns"] == 20
    assert config["vllm"]["max_model_len"] == 65536
    assert config["rollout"]["reasoning_effort"] == "high"
    assert config["rollout"]["max_tokens"] == 8192
    assert config["rollout"]["concurrency"] == 32
    assert config["rejection_sampling"]["concurrency"] == 32
    assert config["meta_evolver"]["generation_concurrency"] == 8
    assert config["evolution"]["max_rounds"] == 20
    guidance = config["meta_evolver"]["curator_guidance"]
    assert "prompt.txt and icl_examples.txt" in guidance
    assert "substantive Python AST change" in guidance
    assert "do not require" in guidance


def test_rbac_uses_remote_lineage_tolerances_and_validation_split():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/rbac.yaml")
    selection = config["selection"]
    assert selection["promotion_split"] == "validation"
    assert selection["safe_ex_tolerance"] == 0.01
    assert selection["ac_f1_tolerance"] == 0.01
    assert selection["policy_validation_tolerance"] == 0.005
    assert selection["max_shortlisted_children"] == 3
    assert selection["primary_quality_metric"] == "safe_ex"
    assert selection["mismatch_metric"] == "interface_mismatch_turn_rate"


def test_domain_model_selection_uses_primary_metric_and_turn_mismatch():
    root = Path(__file__).resolve().parents[1]
    expected_order = {
        "rbac": ["safe_ex", "ac_f1"],
        "homebench": ["em", "f1"],
    }
    for domain, metrics in expected_order.items():
        selection = load_config(root / "configs" / f"{domain}.yaml")["selection"]
        assert selection["primary_quality_metric"] == metrics[0]
        assert selection["quality_metrics"] == metrics
        assert selection["mismatch_metric"] == "interface_mismatch_turn_rate"
        assert selection["mismatch_coverage_metric"] == "complete_interface_turns"
    who_when = load_config(root / "configs/who_when.yaml")["selection"]
    assert "primary_quality_metric" not in who_when
    assert who_when["ranking_metric_groups"] == [["agent_accuracy", "step_accuracy"]]
    assert who_when["ranking_metric_weights"] == {"agent_accuracy": 3, "step_accuracy": 7}
    assert who_when["mismatch_metric"] == "interface_mismatch_turn_rate"


def test_managed_vllm_defaults_are_bounded():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")
    who_when = load_config(Path(__file__).resolve().parents[1] / "configs/who_when.yaml")
    vllm = config["vllm"]
    assert vllm["host"] == "127.0.0.1"
    assert vllm["port"] == 8000
    assert 0 < vllm["gpu_memory_utilization"] < 1
    assert vllm["startup_timeout_seconds"] > 0
    assert vllm["shutdown_timeout_seconds"] > 0
    assert vllm["tensor_parallel_size"] == 1
    assert vllm["data_parallel_size"] == 4
    assert vllm["gpu_memory_utilization"] == 0.8
    assert vllm["extra_args"][-3:] == [
        "qwen3_coder",
        "--reasoning-parser",
        "qwen3",
    ]
    assert who_when["harness"]["max_turns"] == 20
    # The model may be a Hugging Face id or a configured local path.
    assert config["model"]["base"] == "Qwen/Qwen3.5-9B"
    assert config["meta_evolver"]["max_attempts_per_round"] == 8
    assert config["meta_evolver"]["proposal_batch_size"] == 8
    assert config["meta_evolver"]["generation_concurrency"] == 8
    assert config["meta_evolver"]["evidence_mode"] == "trajectory_directory"
    assert config["meta_evolver"]["max_trace_items"] is None
    assert config["meta_evolver"]["codex"]["model"] == "deepseek-flash"
    assert config["meta_evolver"]["codex"]["reasoning_effort"] == "high"
    assert config["meta_evolver"]["codex"]["web_search"] is True
    assert config["rollout"]["reasoning_effort"] == "medium"
    assert config["rollout"]["enable_thinking"] is True
    assert config["rollout"]["attempts_per_sample"] == 5
    assert config["rollout"]["concurrency"] == 16
    assert "evaluation" not in config


def test_domain_configs_resolve_distinct_harness_roots():
    root = Path(__file__).resolve().parents[1]
    who_when = harness_spec(load_config(root / "configs/who_when.yaml"), project_root=root)
    homebench = harness_spec(load_config(root / "configs/homebench.yaml"), project_root=root)
    rbac = harness_spec(load_config(root / "configs/rbac.yaml"), project_root=root)

    assert who_when.kind == "failure_detection"
    assert who_when.package == "harnesses.who_when"
    assert who_when.active_path.is_relative_to(root / "harnesses/who_when")
    assert homebench.kind == "homebench"
    assert homebench.package == "harnesses.homebench"
    assert homebench.path_for(homebench.base) == root / "harnesses/homebench/initial_harness"
    assert homebench.active_path.is_relative_to(homebench.root)
    assert homebench.active_path.is_dir()
    assert len({who_when.root, homebench.root, rbac.root}) == 3


def test_resolved_config_redacts_credentials(tmp_path):
    output = tmp_path / "resolved.yaml"
    write_resolved_config(
        {
            "model": {"api_key": "secret-value", "api_key_env": "MODEL_KEY"},
            "nested": [{"access_token": "token-value"}],
        },
        output,
    )

    text = output.read_text(encoding="utf-8")
    assert "secret-value" not in text
    assert "token-value" not in text
    assert "MODEL_KEY" in text
    assert text.count("[REDACTED_SECRET]") == 2


def test_base_contains_only_shared_sections():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")
    assert set(config) == {
        "meta_evolver",
        "model",
        "inheritance",
        "vllm",
        "evolution",
        "rollout",
        "rejection_sampling",
        "training",
    }
