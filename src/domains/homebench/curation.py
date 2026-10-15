"""HomeBench public task profiles and final-command verification."""

from src.training.curation import final_arguments

from .evaluator import judge


def profile(sample):
    record = sample.record
    category = record.get("metadata", {}).get("category", "unknown")
    return {
        "sample_id": sample.sample_id,
        "question_text": sample.instruction,
        "question": {"instruction": sample.instruction, "home": sample.home},
        "features": {"task_type": record.get("task_type", "unknown"), "category": category},
        "hard": category in {"IM", "MM"},
        "group_keys": [],
    }


def validate_record(record, sample):
    arguments = final_arguments(record)
    commands = arguments.get("commands")
    if (
        not isinstance(commands, str)
        or not judge(commands, sample.record["target"]["commands"])["passed"]
    ):
        raise ValueError("raw final_answer fails official command exact match")
