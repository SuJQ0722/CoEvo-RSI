"""Resumable HomeBench evaluation, with data/model/harness/code provenance."""

import argparse
import hashlib
import json
from pathlib import Path

from src.config import apply_harness_overrides, harness_spec, load_config
from src.domains.io import read_records
from src.runner.execution import ItemProgress, in_completion_order
from src.utils.logging import get_logger
from src.utils.persistence import append_jsonl, atomic_json, atomic_jsonl

from ..client import build_client
from ..data import PROTOCOL, digest, load
from ..evaluator import evaluate, judge
from ..submission import SCORING_VERSION, official_prediction
from .execution import run_sample

LOGGER = get_logger("runtime.homebench.infer")


def identity(samples, config, harness, *, model, base_url, max_turns):
    config = apply_harness_overrides(config, harness_spec(config).path_for(harness))

    def files_digest(root):
        return digest(
            {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*"))
                if p.is_file() and p.suffix in {".py", ".txt", ".yaml", ".json", ".md"}
            }
        )

    return {
        "protocol": PROTOCOL,
        "scoring_version": SCORING_VERSION,
        "harness": harness,
        "harness_digest": files_digest(harness_spec(config).path_for(harness)),
        "runtime_digest": files_digest(Path(__file__).resolve().parents[1]),
        "sample_digests": {s.sample_id: s.digest for s in samples},
        "model": model,
        "model_revision": (config.get("model") or {}).get("revision", "unknown"),
        "model_generation": {
            k: v
            for k, v in (config.get("model") or {}).items()
            if k in {"max_tokens", "temperature", "top_p", "enable_thinking", "reasoning_effort"}
        },
        "base_url": base_url,
        "max_turns": max_turns,
        "prompt_mode": config.get("harness", {}).get("prompt_mode", "icl"),
        "rollout": config.get("rollout", {}),
    }


def collect(
    *,
    config,
    data,
    output_dir,
    model,
    base_url,
    harness,
    max_turns=8,
    concurrency=16,
    limit=None,
    resume=False,
    client_factory=None,
):
    if concurrency < 1 or max_turns < 1:
        raise ValueError("concurrency and max_turns must be positive")
    samples = list(load(data, limit=limit))
    if not samples:
        raise ValueError("empty evaluation dataset")
    config = apply_harness_overrides(config, harness_spec(config).path_for(harness))
    manifest = identity(
        samples, config, harness, model=model, base_url=base_url, max_turns=max_turns
    )
    output_dir = Path(output_dir)
    path, manifest_path = output_dir / "samples.jsonl", output_dir / "run_manifest.json"
    existing = []
    if resume and path.exists():
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("resume refused: data/model/harness/runtime/settings differ")
        existing = list(read_records(path))
        if len({r["sample_id"] for r in existing}) != len(existing):
            raise ValueError("duplicate resume sample IDs")
        for row in existing:
            if row.get("task_digest") != manifest["sample_digests"].get(row["sample_id"]):
                raise ValueError("resume task binding mismatch")
        # Retry infrastructure failures, retain actual model successes/failures.
        existing = [r for r in existing if not r.get("error")]
    atomic_json(manifest_path, manifest)
    completed = {r["sample_id"] for r in existing}
    remaining = [s for s in samples if s.sample_id not in completed]

    LOGGER.info(
        "start data=%s output=%s harness=%s model=%s items=%d reused=%d "
        "max_turns=%d concurrency=%d resume=%s",
        data,
        output_dir,
        harness,
        model,
        len(samples),
        len(existing),
        max_turns,
        concurrency,
        resume,
    )

    def execute(sample):
        client = (
            client_factory(sample)
            if client_factory
            else build_client(config, model=model, base_url=base_url)
        )
        row = run_sample(sample, config, harness, client, max_turns=max_turns)
        return sample, row

    # An interruption can leave a partial JSONL; fail explicitly on malformed
    # resume rather than silently losing records.
    atomic_jsonl(path, existing)
    progress = ItemProgress(LOGGER, total=len(samples), completed=len(existing))
    with path.open("a", encoding="utf-8") as stream:
        for sample, row in in_completion_order(remaining, execute, concurrency):
            append_jsonl(stream, row)
            exact = judge(
                official_prediction(row),
                sample.record["target"]["commands"],
            )["passed"]
            score = bool(row.get("finished") and not row.get("error") and exact)
            progress.log(
                sample_id=sample.sample_id,
                score=score,
                reused=False,
                finished=bool(row.get("finished")),
                error=bool(row.get("error")),
            )
    result = evaluate(path, samples=samples)
    payload = {
        **result.metrics,
        **result.diagnostics,
        "results": result.per_sample,
        "run_manifest": manifest,
    }
    atomic_json(output_dir / "metrics.json", payload)
    LOGGER.info(
        "complete output=%s items=%d em=%.6f f1=%.6f completed=%d errors=%d",
        output_dir,
        len(samples),
        payload["em"],
        payload["f1"],
        payload["completed_count"],
        payload["infrastructure_errors"],
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--harness", default="initial_harness")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    args["config"] = load_config(args["config"])
    collect(**args)


if __name__ == "__main__":
    main()
