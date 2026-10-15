"""RBAC question groups, difficulty, and raw access-label verification."""

from src.training.curation import final_arguments


def profile(sample):
    return {
        "sample_id": sample.sample_id,
        "question_text": sample.instruction,
        "question": {
            "instruction": sample.instruction,
            "test_role": sample.test_role,
            "db_id": sample.db_id,
        },
        "features": {
            "difficulty": sample.difficulty,
            "access_label": str(sample.valid),
            "db_id": sample.db_id,
        },
        "hard": sample.difficulty.lower() in {"hard", "extra", "extra hard", "extra-hard"},
        "group_keys": [sample.question_group_id] if sample.question_group_id else [],
    }


def validate_record(record, sample):
    arguments = final_arguments(record)
    access = arguments.get("access_granted")
    if not isinstance(access, bool) or int(access) != sample.valid:
        raise ValueError("raw final_answer has incorrect access label")
    if access and not str(arguments.get("sql") or "").strip():
        raise ValueError("raw granted final_answer is missing SQL")
