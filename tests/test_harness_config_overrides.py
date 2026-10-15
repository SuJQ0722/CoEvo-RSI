from pathlib import Path

import pytest

from src.config import apply_harness_overrides


ROOT = Path(__file__).resolve().parents[1]


def test_harness_runtime_overrides_cannot_change_data_or_credentials(tmp_path):
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "config.yaml").write_text(
        "overrides:\n  data:\n    splits:\n      val: forbidden.json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported harness overrides"):
        apply_harness_overrides({"data": {}}, harness)
