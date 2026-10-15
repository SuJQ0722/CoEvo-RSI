"""Per-round metadata persistence for the meta-agent."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def save_round_metadata(
    round_dir: Path,
    *,
    round_number: int,
    child_number: int,
    parent_name: str,
    harness_name: str,
    strategy: str,
    hypothesis: str,
    mutations_applied: list[str],
    evaluation_config: dict[str, Any] | None = None,
) -> Path:
    """Save structured metadata about this evolution round."""
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "round": round_number,
        "child": child_number,
        "parent_harness": parent_name,
        "child_harness": harness_name,
        "strategy": strategy,
        "hypothesis": hypothesis,
        "mutations_applied": mutations_applied,
        "evaluation_config": evaluation_config or {},
    }
    path = round_dir / "metadata.json"
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
