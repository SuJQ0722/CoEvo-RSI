from __future__ import annotations

import json

from src.domains.rbac.evidence import build


def test_build_uses_six_category_per_sample_result(tmp_path):
    source = tmp_path / "samples.jsonl"
    source.write_text(
        json.dumps(
            {
                "sample_id": "denied-correctly",
                "prediction": {"access": 0},
                "gold": {"valid": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = build(source, tmp_path / "evidence.jsonl")

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["outcome"]["metrics"]["score"] == 1
    assert record["domain_payload"]["predicted_access"] == 0
