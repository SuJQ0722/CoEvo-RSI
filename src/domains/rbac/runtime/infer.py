"""Run a rule-first RBAC harness with a generative-LLM fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import active_harness, load_config
from src.diagnostics.observe import ObservedModel
from src.domains.rbac.data import RBACSample, load, load_policy_rows
from src.domains.rbac.evaluator import score_record
from src.runner.execution import ItemProgress, in_completion_order
from src.runner.model_client import OpenAICompatibleClient
from src.utils.logging import get_logger
from src.utils.persistence import append_jsonl
from harnesses.rbac.initial_harness.tools.final_answer import marker
from harnesses.rbac.loader import build_harness

LOGGER = get_logger("runtime.rbac.infer")


def _project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _resolve_db_path(db_dir: str | None, db_id: str) -> str | None:
    from src.domains.rbac.databases import resolve_db_path

    return resolve_db_path(db_dir, db_id)


def evaluate_one(
    sample: RBACSample, harness: Any, model: Any, *, db_path: str | None = None
) -> dict[str, Any]:
    route = harness.route(sample.instruction, sample.test_role)
    model = ObservedModel(model)
    rule_used = route.decision is not None
    messages: list[dict[str, Any]] = []
    response = marker(bool(route.decision)) if rule_used else "ACCESS_DENIED"
    usage: dict[str, int] = {}
    error = ""
    llm_called = not rule_used
    latency = 0.0
    sql: str | None = None
    turns = 0
    if llm_called:
        result = harness.call_llm(
            model,
            sample.instruction,
            sample.test_role,
            route,
            db_path=db_path,
        )
        messages = result.messages
        response = result.response
        usage = result.usage
        error = result.error
        latency = result.latency_seconds
        sql = getattr(result, "sql", None)
        turns = getattr(result, "turns", 0)
    if rule_used:
        predicted = int(route.decision)
    elif error:
        predicted = None
    else:
        predicted = harness.parse_access(response)

    gold_sql = getattr(sample, "gold_sql", None)

    return {
        "sample_id": sample.sample_id,
        "instruction": sample.instruction,
        "test_role": sample.test_role,
        "test_type": sample.test_type,
        "db_id": sample.db_id,
        "difficulty": sample.difficulty,
        "question_group_id": sample.question_group_id,
        "source_split": sample.source_split,
        "messages": messages,
        "prediction": {
            "access": int(predicted) if predicted is not None else None,
            "response": response,
            "sql": sql,
            "turns": turns,
            "route_source": route.source,
            "rule_confidence": float(route.confidence),
            "rule_decision": int(route.decision) if route.decision is not None else None,
        },
        "gold": {
            "valid": int(sample.valid),
            "sql": gold_sql,
        },
        "db_path": db_path,
        "llm_called": int(llm_called),
        "rule_used": int(rule_used),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "token_usage_available": rule_used or bool(usage),
        "latency_seconds": latency,
        "error": error,
        "interface_trace": model.trace(
            "rule_only"
            if rule_used
            else "provider_error"
            if any("error_type" in r for r in model.requests)
            else "budget_exhausted"
            if "exhausted" in error and "turns" in error
            else "runtime_error"
            if error
            else "submitted"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rbac.yaml"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--harness")
    parser.add_argument("--db-dir", type=str, help="Path to Spider database directory")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = Path(__file__).resolve().parents[4]
    train_value = args.train_data or (config.get("data") or {}).get("splits", {}).get("train")
    if not train_value:
        parser.error("RBAC training split is required to compile the policy index")
    train_path = _project_path(train_value, project_root)
    harness_name = args.harness or active_harness(config)
    harness = build_harness(harness_name, load_policy_rows(train_path))
    rollout = dict(config.get("rollout") or {})
    concurrency = args.concurrency or int(rollout.get("concurrency", 1))

    db_dir = args.db_dir
    if not db_dir:
        # Try to find Spider databases in standard locations
        for candidate in ["data/spider/database", "data/spider"]:
            p = project_root / candidate
            if p.is_dir():
                db_dir = str(p)
                break

    model = OpenAICompatibleClient(
        seed=rollout.get("seed"),
        model=args.model,
        base_url=args.base_url,
        temperature=float(rollout.get("temperature", 0.0)),
        top_p=float(rollout.get("top_p", 1.0)),
        top_k=rollout.get("top_k"),
        max_tokens=int(rollout.get("max_tokens", 512)),
        reasoning_effort=rollout.get("reasoning_effort"),
        enable_thinking=rollout.get("enable_thinking", False),
        timeout=float(rollout.get("timeout_seconds", 300)),
    )
    samples = list(load(args.data, limit=args.limit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reusable: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.is_file():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("sample_id"):
                reusable[str(record["sample_id"])] = record

    def work(sample: RBACSample) -> tuple[dict[str, Any], bool]:
        if sample.sample_id in reusable:
            return reusable[sample.sample_id], True
        sample_db_path = _resolve_db_path(db_dir, sample.db_id)
        return evaluate_one(sample, harness, model, db_path=sample_db_path), False

    progress = ItemProgress(LOGGER, total=len(samples))
    with args.output.open("w", encoding="utf-8") as stream:
        for record, reused in in_completion_order(samples, work, concurrency):
            append_jsonl(stream, record)
            progress.log(
                sample_id=record["sample_id"],
                score=score_record(record),
                reused=reused,
                llm_called=bool(record.get("llm_called")),
                error=bool(record.get("error")),
            )


if __name__ == "__main__":
    main()
