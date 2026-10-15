"""Who&When difficulty, public questions, and raw-answer acceptance for SFT."""

from types import SimpleNamespace

from src.traces import DetectionResult
from src.training.curation import final_arguments

from .rejection_sampling import _difficulty_features, accept_rollout


def profile(sample):
    length, position, agents = _difficulty_features(sample)
    return {
        "sample_id": sample.sample_id,
        "question_text": sample.task,
        "question": {"task": sample.task, "trajectory": [step.raw for step in sample.steps]},
        "features": {
            "trace_length": length,
            "decisive_position": position,
            "agent_count": agents,
            "source": sample.source_dataset,
        },
        "hard": len(sample.steps) > 20 or agents in {"4", "5+"} or position == "00_0-25%",
        "group_keys": [
            str(sample.metadata[k])
            for k in ("task_id", "question_group_id")
            if sample.metadata.get(k)
        ],
    }


def validate_record(record, sample):
    arguments = final_arguments(record)
    # Check the model's submitted order, before any harness-side promotion.
    result = DetectionResult(
        is_failure=arguments.get("is_failure"),
        evidence_step_ids=tuple(arguments.get("evidence_step_ids") or ()),
        responsible_agent=arguments.get("responsible_agent") or "",
    )
    rollout = SimpleNamespace(
        result=result, inspected_step_ids=record.get("sampling", {}).get("inspected_step_ids", [])
    )
    if not isinstance(result.is_failure, bool) or not accept_rollout(rollout, sample):
        raise ValueError("raw final_answer does not satisfy native labels/inspection")
