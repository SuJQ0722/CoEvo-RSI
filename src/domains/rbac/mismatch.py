"""RBAC separates model protocol, SQL execution and database availability."""

from src.diagnostics.mismatch import MismatchContract


class RBACMismatch(MismatchContract):
    min_tool_calls = 1
    terminal_tool = "final_answer"

    def state(self, record):
        prediction = record.get("prediction") or {}
        rule = bool(record.get("rule_used")) and not record.get("llm_called")
        error = str(record.get("error") or "")
        finished = not error and prediction.get("access") in (0, 1)
        invalid_sql_field = prediction.get("access") == 1 and not (
            isinstance(prediction.get("sql"), str) and prediction["sql"].strip()
        )
        return {
            "rule_only": rule,
            "finished": finished if "prediction" in record else None,
            "final_format_error": bool(invalid_sql_field) if finished and not rule else None,
            "termination": "rule_only"
            if rule
            else "submitted"
            if finished
            else "budget_exhausted"
            if "exhausted" in error and "turns" in error
            else "unknown",
        }

    def tool_return_errors(self, name, payload):
        if not isinstance(payload, dict):
            return ["return_protocol"] if name in {"test_sql", "final_answer"} else None
        error = str(payload.get("error") or "")
        if error:
            if "no database available" in error or any(
                s in error
                for s in ("unable to open database", "database is locked", "disk I/O error")
            ):
                return ["infrastructure"]
            if "unknown tool" in error:
                return ["protocol"]
            if name == "test_sql":
                return ["argument_semantic"] if "sql must be" in error else ["execution"]
            if name == "final_answer" and "access_granted must be" in error:
                return ["argument_semantic"]
            return ["unknown"]
        if name == "final_answer":
            return [] if payload.get("status") == "submitted" else ["return_protocol"]
        if name == "test_sql":
            return [] if isinstance(payload.get("rows"), list) else ["return_protocol"]
        return None

    def task_correct(self, result):
        return bool(result["score"]) if "score" in result else None
