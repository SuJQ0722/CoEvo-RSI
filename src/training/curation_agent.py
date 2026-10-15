"""Data-curation agent with controller-owned tools, without host code execution.

The model receives only staged evidence and can change only a record-count plan.
No shell, arbitrary file paths, Python execution or additional endpoints are tools.
This boundary works in GPU containers without Landlock or user namespaces.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from copy import deepcopy

from jsonschema import validate

from src.runner.model_client import OpenAICompatibleClient
from src.utils.logging import get_logger
from src.utils.persistence import atomic_json

LOGGER = get_logger("training.curation_agent")


def tool(name, description, properties, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


FILTER = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "hard": {"type": "boolean"},
        "features": {"type": "object", "additionalProperties": {"type": "string"}},
        "record_ids": {"type": "array", "items": {"type": "string"}},
        "min_assistant_characters": {"type": "integer", "minimum": 0},
        "max_assistant_characters": {"type": "integer", "minimum": 0},
    },
}
TOOLS = [
    tool(
        "inspect_catalog",
        "Filter the catalog and return paginated records plus distributions. "
        "An empty filter selects all records. Feature filters use exact observed values.",
        {
            "filter": FILTER,
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 30},
        },
    ),
    tool(
        "read_data",
        "Read a bounded byte range from one supplied evidence file. "
        "Continue with next_offset to inspect long trajectories or validation questions.",
        {
            "file": {
                "type": "string",
                "enum": ["catalog.jsonl", "records.jsonl", "val_questions.jsonl"],
            },
            "offset": {"type": "integer", "minimum": 0},
            "length": {"type": "integer", "minimum": 1, "maximum": 16000},
        },
        ["file"],
    ),
    tool(
        "set_counts",
        "Set physical repetition counts for records matching a filter. "
        "Unchanged records default to one. All hard records must remain; only hard records "
        "may be repeated. Returns resulting mixture statistics.",
        {
            "filter": FILTER,
            "count": {"type": "integer", "minimum": 0},
        },
        ["filter", "count"],
    ),
    tool(
        "submit_plan",
        "Validate and submit the current mixture. Validation errors are returned "
        "for correction. Submit only after inspecting evidence and choosing the mixture.",
        {
            "rationale": {"type": "string", "minLength": 1},
        },
        ["rationale"],
    ),
]
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}


class CurationTools:
    def __init__(self, files, inputs):
        self.files = files
        self.inputs = inputs
        with files["catalog.jsonl"].open() as stream:
            self.catalog = [json.loads(line) for line in stream if line.strip()]
        self.counts = {}
        self.plan = None

    def matching(self, criteria):
        ids = criteria.get("record_ids")
        known = {row["record_id"] for row in self.catalog}
        if ids is not None and set(ids) - known:
            raise ValueError("unknown record IDs")
        features = criteria.get("features", {})
        if set(features) - {key for row in self.catalog for key in row["features"]}:
            raise ValueError("unknown feature name; inspect catalog distributions first")
        return [
            row
            for row in self.catalog
            if (ids is None or row["record_id"] in ids)
            and ("hard" not in criteria or row["hard"] == criteria["hard"])
            and all(str(row["features"].get(k)) == v for k, v in features.items())
            and row["assistant_characters"] >= criteria.get("min_assistant_characters", 0)
            and row["assistant_characters"]
            <= criteria.get("max_assistant_characters", float("inf"))
        ]

    def summary(self, rows, *, weighted=False):
        features = {}
        total = hard = characters = 0
        samples, records = set(), set()
        for row in rows:
            count = self.counts.get(row["record_id"], 1) if weighted else 1
            if count:
                records.add(row["record_id"])
                samples.add(row.get("sample_id", row["record_id"]))
            total += count
            hard += count * row["hard"]
            characters += count * row["assistant_characters"]
            for key, value in row["features"].items():
                features.setdefault(key, Counter())[str(value)] += count
        return {
            "records": total,
            "unique_records": len(records),
            "unique_samples": len(samples),
            "repeated_records": total - len(records),
            "hard_records": hard,
            "features": features,
            "assistant_characters": characters,
        }

    def call(self, name, args):
        if name not in SCHEMAS:
            raise ValueError(f"unknown tool: {name}")
        validate(args, SCHEMAS[name])
        if name == "read_data":
            offset = args.get("offset", 0)
            with self.files[args["file"]].open("rb") as stream:
                stream.seek(offset)
                chunk = stream.read(args.get("length", 8000))
                eof = not stream.read(1)
            return {
                "text": chunk.decode("utf-8", errors="replace"),
                "next_offset": offset + len(chunk),
                "eof": eof,
            }
        if name == "inspect_catalog":
            rows = self.matching(args.get("filter", {}))
            offset, limit = args.get("offset", 0), args.get("limit", 10)
            return {
                "summary": self.summary(rows),
                "rows": [
                    dict(row, question=str(row.get("question", ""))[:1200])
                    for row in rows[offset : offset + limit]
                ],
                "next_offset": offset + limit if offset + limit < len(rows) else None,
            }
        if name == "set_counts":
            rows = self.matching(args["filter"])
            if not rows:
                raise ValueError("filter matched no records")
            count = args["count"]
            if count > self.inputs["max_records"]:
                raise ValueError("count exceeds maximum dataset size")
            if any(row["hard"] and count == 0 for row in rows):
                raise ValueError("protected hard record was dropped")
            if any(not row["hard"] and count > 1 for row in rows):
                raise ValueError("only hard records may be repeated")
            self.counts.update({row["record_id"]: count for row in rows})
            return {"changed": len(rows), "mixture": self.summary(self.catalog, weighted=True)}
        total = self.summary(self.catalog, weighted=True)["records"]
        if not self.inputs["min_records"] <= total <= self.inputs["max_records"]:
            raise ValueError(
                f"physical record count {total} is outside "
                f"[{self.inputs['min_records']}, {self.inputs['max_records']}]"
            )
        self.plan = {
            "default_count": 1,
            "counts": dict(self.counts),
            "rationale": args["rationale"],
        }
        return {"submitted": True, "records": total}


def build_client(config):
    meta = config["meta_evolver"]
    backend = meta.get("backend", "codex")
    if backend != "codex":
        raise ValueError("curation tools require the configured OpenAI-compatible meta provider")
    settings = meta[backend]
    base_url = settings.get("base_url")
    model = settings.get("model")
    if not base_url or not model:
        raise ValueError("curation tools require meta_evolver.codex.base_url and model")
    curation = config.get("training", {}).get("data_curation", {})
    max_tokens = int(curation.get("agent_max_tokens", 16384))
    if max_tokens < 1:
        raise ValueError("data_curation.agent_max_tokens must be positive")
    return OpenAICompatibleClient(
        model=model,
        base_url=base_url,
        api_key=settings.get("api_key")
        or os.environ.get(settings.get("api_key_env", "OPENAI_API_KEY")),
        api_mode="responses",  # Same wire protocol as the configured Codex provider.
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=None,
        top_k=None,
        reasoning_effort=settings.get("reasoning_effort", "high"),
        enable_thinking=None,
        tool_choice="auto",
        timeout=300,
    )


def run_curation_agent(*, config, output_dir, inputs, input_files):
    """Execute only data tools; the model never launches a host process."""
    from jsonschema import ValidationError

    from .curation import PROMPT

    settings = config.get("training", {}).get("data_curation", {})
    limit = int(settings.get("agent_max_turns", 32))
    if limit < 1:
        raise ValueError("data_curation.agent_max_turns must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "inputs.json", inputs)
    session = CurationTools(input_files, inputs)
    client = build_client(config)
    token_cap = int(settings.get("agent_max_tokens_cap", max(32768, client.max_tokens or 0)))
    if token_cap < (client.max_tokens or 0):
        raise ValueError("data_curation.agent_max_tokens_cap must be >= agent_max_tokens")
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": json.dumps(inputs, ensure_ascii=False)},
    ]
    events = []
    for turn in range(1, limit + 1):
        response = client.complete(deepcopy(messages), deepcopy(TOOLS))
        events.append({"turn": turn, "response": response})
        atomic_json(output_dir / "events.json", events)
        budget = client.max_tokens
        used = (response.get("_coevox_usage") or {}).get("completion_tokens", 0)
        truncated = response.get("_coevox_incomplete_reason") == "max_output_tokens"
        if truncated or (not response.get("tool_calls") and budget and used >= budget):
            if not budget or budget >= token_cap:
                raise RuntimeError(f"data-curation agent exhausted output token budget {budget}")
            client.max_tokens = min(2 * budget, token_cap)
            events[-1]["recovery"] = {"max_tokens": client.max_tokens}
            atomic_json(output_dir / "events.json", events)
            LOGGER.warning(
                "curation turn=%d exhausted output budget=%d; retrying with %d",
                turn, budget, client.max_tokens,
            )
            messages.append({
                "role": "user",
                "content": "Your last response exhausted its output budget before completing. "
                "The budget has been increased. Take a small concrete next step with the data "
                "tools; use their statistics instead of calculating the whole mixture mentally.",
            })
            continue
        message = {
            k: v
            for k, v in response.items()
            if k in {"role", "content", "tool_calls", "reasoning_content"}
        }
        message["role"] = "assistant"
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            messages.append(
                {
                    "role": "user",
                    "content": "No plan was submitted. Use the data tools "
                    "and finish with submit_plan; plain text alone does not complete this task.",
                }
            )
            continue
        for call in calls:
            try:
                result = session.call(
                    call["function"]["name"], json.loads(call["function"]["arguments"])
                )
            except (ValueError, TypeError, KeyError, ValidationError) as exc:
                result = {"error": str(exc)[:3000]}
            events[-1].setdefault("tools", []).append({"call_id": call["id"], "result": result})
            atomic_json(output_dir / "events.json", events)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
            if session.plan is not None:
                workspace = output_dir / "workspace"
                atomic_json(workspace / "plan.json", session.plan)
                result_path = workspace / "final_message.md"
                atomic_json(
                    result_path, {"plan_file": "plan.json", "rationale": session.plan["rationale"]}
                )
                return result_path
    raise RuntimeError(f"data-curation agent did not submit a valid plan in {limit} turns")
