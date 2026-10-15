"""HomeBench tool envelopes and the official final submission transport."""

import re

from src.diagnostics.mismatch import MismatchContract

from .submission import official_prediction


class HomeBenchMismatch(MismatchContract):
    min_tool_calls = max_tool_calls = 1
    terminal_tool = "final_answer"

    def state(self, record):
        finished = bool(record.get("finished")) if "finished" in record else None
        invalid = None
        if finished:
            try:
                text = official_prediction(record).strip()
                invalid = not bool(re.fullmatch(r"\{[^{}]+\}(?:\s*\{[^{}]+\})*", text))
            except (ValueError, TypeError, KeyError):
                invalid = True
        return {
            "finished": finished,
            "final_format_error": invalid,
            "termination": "submitted" if finished else "unknown",
        }

    def tool_return_errors(self, name, payload):
        if isinstance(payload, dict) and payload.get("error"):
            error = str(payload["error"])
            if error == "unknown tool":
                return ["protocol"]
            if error in {
                "invalid JSON object arguments",
                "room must be a string",
                "room/device must be strings",
            }:
                return ["argument_semantic"]
            return ["unknown"]
        if name == "inspect_home":
            if not isinstance(payload, dict):
                return ["return_protocol"]
            # Only the room envelope, never nested device state or text.
            return (
                ["argument_semantic"]
                if any(
                    isinstance(room, dict) and room.get("error") == "unknown room"
                    for room in payload.values()
                )
                else []
            )
        if name == "list_methods":
            return [] if isinstance(payload, list) else ["return_protocol"]
        return None

    def task_correct(self, result):
        return bool(result["passed"]) if "passed" in result else None
