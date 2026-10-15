"""Who&When interface and observation semantics for passive diagnostics."""

from src.diagnostics.mismatch import MismatchContract


class WhoWhenMismatch(MismatchContract):
    min_tool_calls = max_tool_calls = 1
    terminal_tool = "final_answer"

    def state(self, record):
        error = str(record.get("error") or "")
        return {
            "finished": bool(record.get("prediction")) if "prediction" in record else None,
            "final_format_error": False if record.get("prediction") else None,
            "termination": "submitted"
            if record.get("prediction")
            else "budget_exhausted"
            if "max turns reached" in error
            else "protocol_error"
            if "no tool call" in error
            else "unknown",
        }

    def tool_return_errors(self, name, payload):
        if not isinstance(payload, dict):
            return ["return_protocol"] if name in {"inspect", "final_answer"} else None
        if payload.get("error"):
            error = str(payload["error"])
            if "Only one tool call" in error or "unknown tool" in error:
                return ["protocol"]
            if error in {
                "step_id_list must be a non-empty list",
                "is_failure must be a boolean",
                "final_answer is missing required attribution fields",
            } or error.startswith("Too many steps ("):
                return ["argument_semantic"]
            return ["unknown"]
        if name == "inspect":
            steps = payload.get("steps")
            if isinstance(steps, list):
                errors = [
                    str(step["error"])
                    for step in steps
                    if isinstance(step, dict) and step.get("error")
                ]
                return sorted(
                    {
                        "argument_semantic"
                        if error in {"invalid step ID", "step not found"}
                        else "unknown"
                        for error in errors
                    }
                )
            return [] if payload.get("truncated") else ["return_protocol"]
        if name == "final_answer":
            return [] if payload.get("status") == "submitted" else ["return_protocol"]
        return None
