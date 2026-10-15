"""Per-round report generation.

Each time the meta-agent generates a new harness, a report is saved alongside
it documenting what changed, why, and key metrics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def generate_round_report(
    round_dir: Path,
    *,
    round_number: int,
    child_number: int,
    parent_name: str,
    changes_summary: str,
    meta_reasoning: dict[str, Any] | None = None,
    metrics_before: dict[str, Any] | None = None,
    metrics_after: dict[str, Any] | None = None,
) -> Path:
    """Generate a report.json in the round directory."""
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "round": round_number,
        "child": child_number,
        "parent_harness": parent_name,
        "changes_summary": changes_summary,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
    }
    report_path = round_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if meta_reasoning:
        save_meta_reasoning(round_dir, meta_reasoning)

    return report_path


def save_meta_reasoning(round_dir: Path, reasoning: dict[str, Any]) -> Path:
    """Save the meta-agent's reasoning/thought process as JSON."""
    meta_path = round_dir / "meta_reasoning.json"
    meta_path.write_text(
        json.dumps(reasoning, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return meta_path


def make_round_dirname(
    round_number: int,
    child_number: int,
    timestamp: datetime | None = None,
) -> str:
    """Generate directory name: round_N_child_n_YYYYMMDDTHHMMSSZ."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    ts = timestamp.strftime("%Y%m%dT%H%M%SZ")
    return f"round_{round_number}_child_{child_number}_{ts}"
