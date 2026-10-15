"""Collect failure-detection rollouts from a JSONL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import active_harness, harness_spec, load_config
from src.domains import get_domain_plugin
from src.domains.who_when.data import WhoWhenDataLoader
from src.domains.who_when.runtime.evaluation import StrictAttributionJudge
from src.domains.who_when.runtime.rollout import Rollout, run_rollout
from src.runner.execution import ItemProgress, in_completion_order
from src.runner.model_client import ChatModel, OpenAICompatibleClient
from src.traces import AgentTrace
from src.utils.logging import get_logger
from src.utils.persistence import append_jsonl

LOGGER = get_logger("runtime.who_when.infer")

# Compatibility for callers that imported the old domain-private helper.
_in_completion_order = in_completion_order


def iter_traces(path: Path, limit: int | None = None):
    """Backward-compatible Who&When iterator."""

    yield from WhoWhenDataLoader(path, limit)


def _prediction_is_complete(record: dict[str, Any]) -> bool:
    prediction = record.get("prediction")
    if not isinstance(prediction, dict) or prediction.get("is_failure") is None:
        return False
    gold = record.get("gold") or {}
    if gold.get("decisive_step") is not None and not prediction.get("evidence_step_ids"):
        return False
    return not (
        str(gold.get("responsible_agent") or "").strip()
        and not str(prediction.get("responsible_agent") or "").strip()
    )


def _load_resumable_records(path: Path) -> dict[str, dict[str, Any]]:
    """Load successful records, ignoring malformed tails left by interruption."""
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("resume ignore malformed record path=%s line=%d", path, line_number)
                continue
            sample_id = str(record.get("sample_id") or "")
            if sample_id and _prediction_is_complete(record):
                records[sample_id] = record
    return records


def run_rollout_with_retries(
    trace: AgentTrace,
    model: ChatModel,
    *,
    max_turns: int,
    harness_name: str,
    attempts: int,
) -> Rollout:
    """Retry an entire rollout until it produces a parseable attribution."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    last: Rollout | None = None
    for attempt in range(1, attempts + 1):
        last = run_rollout(trace, model, max_turns, harness_name=harness_name)
        if _prediction_is_complete(last.to_record(trace)):
            return last
        if attempt < attempts:
            LOGGER.warning(
                "retry sample_id=%s attempt=%d/%d error=%s",
                trace.sample_id,
                attempt,
                attempts,
                last.error or "incomplete attribution",
            )
    assert last is not None
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/who_when.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--request-timeout", type=float)
    parser.add_argument("--attempts-per-sample", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--harness")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete predictions already present in the output JSONL",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    selected_harness = harness_spec(
        config,
        project_root=Path(__file__).resolve().parents[4],
    )
    domain = get_domain_plugin(config)
    if selected_harness.kind != "failure_detection":
        parser.error(
            "runner.infer is the Who&When trajectory evaluator; "
            f"domain={selected_harness.domain} must use its own sampling runner"
        )
    harness_name = args.harness or active_harness(config)
    rollout = dict(config.get("rollout") or {})
    attempts_per_sample = int(
        args.attempts_per_sample
        if args.attempts_per_sample is not None
        else rollout.get("attempts_per_sample", 3)
    )
    if attempts_per_sample < 1:
        parser.error("--attempts-per-sample must be positive")
    concurrency = int(
        args.concurrency if args.concurrency is not None else rollout.get("concurrency", 1)
    )
    if concurrency < 1:
        parser.error("--concurrency must be positive")

    def configured(name: str, cli_value, default):
        return cli_value if cli_value is not None else rollout.get(name, default)

    client = OpenAICompatibleClient(
        seed=(config.get("rollout") or {}).get("seed"),
        model=args.model,
        base_url=args.base_url,
        temperature=float(configured("temperature", args.temperature, 0.6)),
        top_p=float(configured("top_p", args.top_p, 0.95)),
        top_k=int(configured("top_k", args.top_k, 20)),
        max_tokens=int(configured("max_tokens", args.max_tokens, 4096)),
        reasoning_effort=str(configured("reasoning_effort", args.reasoning_effort, "medium")),
        enable_thinking=bool(configured("enable_thinking", args.enable_thinking, True)),
        timeout=float(configured("timeout_seconds", args.request_timeout, 300)),
    )
    LOGGER.info(
        "start data=%s harness=%s limit=%s reasoning_effort=%s thinking=%s attempts=%d concurrency=%d resume=%s",
        args.data,
        harness_name,
        args.limit,
        client.reasoning_effort,
        client.enable_thinking,
        attempts_per_sample,
        concurrency,
        args.resume,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reusable = _load_resumable_records(args.output) if args.resume else {}

    def evaluate_trace(trace: AgentTrace) -> tuple[AgentTrace, dict[str, Any], bool]:
        reused = trace.sample_id in reusable
        record = reusable.get(trace.sample_id)
        reused = record is not None
        if record is None:
            result = run_rollout_with_retries(
                trace,
                client,
                max_turns=args.max_turns,
                harness_name=harness_name,
                attempts=attempts_per_sample,
            )
            record = result.to_record(trace)
            record["attempts_per_sample"] = attempts_per_sample
            record["retry_exhausted"] = not _prediction_is_complete(record)
        return trace, record, reused

    traces = list(domain.load_data(args.data, limit=args.limit))
    progress = ItemProgress(LOGGER, total=len(traces))
    judge = StrictAttributionJudge()
    with args.output.open("w", encoding="utf-8") as output:
        completed = in_completion_order(traces, evaluate_trace, concurrency)
        for trace, record, reused in completed:
            append_jsonl(output, record)
            decision = judge.judge(record)
            agent_labeled = bool(str((record.get("gold") or {}).get("responsible_agent") or ""))
            score = bool(
                not record.get("error")
                and decision.step_correct
                and (not agent_labeled or decision.agent_correct)
            )
            progress.log(
                sample_id=trace.sample_id,
                score=score,
                reused=reused,
                prediction_complete=_prediction_is_complete(record),
                error=bool(record.get("error")),
            )
    LOGGER.info("complete output=%s rollouts=%d", args.output, progress.completed)


if __name__ == "__main__":
    main()
