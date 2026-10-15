"""Serialize a verified final_answer tool payload into the official text protocol.

This is transport adaptation, not correction of command names, values or gold.
Raw model messages and generated remain untouched. Bare free-text responses and
unfinished reasoning are never promoted to successful tool submissions.
"""

import json

SCORING_VERSION = "homebench_tool_submission_v2"


def compact_record(record):
    """Keep only the final tool message needed to verify a submission."""
    return {
        **{
            k: record.get(k) for k in ("sample_id", "task_digest", "error", "generated", "finished")
        },
        "messages": [
            {"role": m.get("role"), "tool_calls": m.get("tool_calls", [])}
            for m in (record.get("messages") or [])[-1:]
        ],
    }


def official_prediction(record):
    text = record.get("generated") or ""
    if not record.get("finished") or record.get("error"):
        return text
    last = (record.get("messages") or [{}])[-1]
    calls = last.get("tool_calls") or []
    if last.get("role") != "assistant" or len(calls) != 1:
        raise ValueError("finished prediction lacks its final tool submission")
    function = calls[0].get("function") or {}
    args = function.get("arguments")
    args = json.loads(args) if isinstance(args, str) else args
    if (
        function.get("name") != "final_answer"
        or not isinstance(args, dict)
        or set(args) != {"commands"}
        or not isinstance(args["commands"], str)
        or args["commands"] != text
    ):
        raise ValueError("prediction does not match its final_answer.commands")
    # Preserve already bracketed output, including malformed braces: the fixed
    # official parser decides its score. Do not strip prose or alter commands.
    if text.strip() and "{" not in text and "}" not in text:
        return "{" + text + "}"
    return text
