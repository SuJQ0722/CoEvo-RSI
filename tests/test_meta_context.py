import json
import subprocess

import pytest
from jsonschema import ValidationError
from jsonschema import validate as validate_json_schema

from src.meta_evolver.context import (
    _normalize_brief_references,
    _prune_forbidden_properties,
    briefs_schema,
)


def test_prune_forbidden_properties_recurses_through_arrays():
    output = {
        "briefs": [
            {
                "hypothesis": "h",
                "reference_ids": ["r"],
                "evidence_refs": ["e"],
                "constraints": ["c"],
                "expected_effect": "effect",
                "acceptance_checks": ["check"],
                "reference_ids_note": None,
            }
        ],
        "presentation_note": "ignore me",
    }
    schema = briefs_schema(1)

    removed = _prune_forbidden_properties(output, schema)

    assert removed == [("presentation_note",), ("briefs", 0, "reference_ids_note")]
    validate_json_schema(output, schema)


def test_prune_forbidden_properties_does_not_mask_semantic_errors():
    output = {"briefs": [{"hypothesis": 3, "extra": None}]}
    schema = briefs_schema(1)

    assert _prune_forbidden_properties(output, schema) == [("briefs", 0, "extra")]
    with pytest.raises(ValidationError):
        validate_json_schema(output, schema)


def test_normalize_brief_references_preserves_evidence_and_maps_harnesses():
    incumbent = "round_01_retry_04"
    history = "round_01"
    context = {
        # Legacy saved packets do not have incumbent_harness_id, so the first
        # reference remains the resume-compatible fallback.
        "references": [
            {"harness_id": incumbent},
            {"harness_id": history},
        ]
    }
    brief = {
        "reference_ids": [
            "source:incumbent:guard",
            "validation:incorrect-grants",
            f"history:{history}:score-card",
            "domain-contract:output-schema",
            history,
        ]
    }

    _normalize_brief_references(brief, context)

    assert brief["reference_ids"] == [incumbent, history]
    assert brief["evidence_refs"] == [
        "source:incumbent:guard",
        "validation:incorrect-grants",
        f"history:{history}:score-card",
        "domain-contract:output-schema",
    ]


def test_normalize_brief_references_grounds_unknown_label_in_incumbent():
    context = {
        "incumbent_harness_id": "incumbent",
        "references": [{"harness_id": "incumbent"}],
    }
    brief = {"reference_ids": ["unavailable-harness"]}

    _normalize_brief_references(brief, context)

    assert brief["reference_ids"] == ["incumbent"]
    assert brief["evidence_refs"] == ["unavailable-harness"]


@pytest.mark.parametrize("empty_file", [True, False])
def test_empty_meta_output_retries_then_caches_valid_answer(tmp_path, monkeypatch, empty_file):
    from src.meta_evolver import context

    calls = []
    output = tmp_path / "stage"
    workspace = output / "workspace"
    workspace.mkdir(parents=True)
    result = workspace / "final_message.md"
    result.write_text('{"answer":"stale interrupted output"}')
    monkeypatch.setattr(context, "_require_command", lambda command: None)

    def run(command, **kwargs):
        assert not result.exists()
        calls.append(kwargs["input"])
        if len(calls) == 1:
            if empty_file:
                result.write_text("")
        else:
            result.write_text('{"answer":"valid"}')
        return subprocess.CompletedProcess(command, 0, '{"type":"turn.completed"}\n', "")

    monkeypatch.setattr(context.subprocess, "run", run)
    kwargs = {
        "config": {"meta_evolver": {"backend": "codex", "codex": {"command": "fake"}}},
        "project_root": tmp_path,
        "output_dir": output,
        "inputs": {"evidence": "frozen"},
        "prompt": "Return the answer",
        "output_schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    }
    assert json.loads(context.run_meta_text(**kwargs).read_text()) == {"answer": "valid"}
    assert len(calls) == 2
    assert "without a final response" in calls[1]
    assert (output / "attempt_01.stdout.log").exists()
    assert json.loads((output / "complete.json").read_text())["attempts"] == 2
    assert context.run_meta_text(**kwargs) == result
    assert len(calls) == 2  # Resume reuses the completed result without another call.


def test_empty_meta_output_exhaustion_is_not_cached(tmp_path, monkeypatch):
    from src.meta_evolver import context

    calls = []
    monkeypatch.setattr(context, "_require_command", lambda command: None)

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, '{"type":"turn.completed"}\n', "")

    monkeypatch.setattr(context.subprocess, "run", run)
    kwargs = {
        "config": {"meta_evolver": {"backend": "codex", "codex": {"command": "fake"}}},
        "project_root": tmp_path,
        "output_dir": tmp_path / "stage",
        "inputs": {},
        "prompt": "Return a result",
    }
    with pytest.raises(RuntimeError, match="empty MetaAgent output after 3 attempts"):
        context.run_meta_text(**kwargs)
    assert len(calls) == 3
    assert not (kwargs["output_dir"] / "complete.json").exists()

    def recovered(command, **kwargs):
        (kwargs["cwd"] / "final_message.md").write_text("recovered")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(context.subprocess, "run", recovered)
    assert context.run_meta_text(**kwargs).read_text() == "recovered"


def test_meta_process_failure_is_not_treated_as_empty_success(tmp_path, monkeypatch):
    from src.meta_evolver import context

    calls = []
    monkeypatch.setattr(context, "_require_command", lambda command: None)

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, "", "provider failure")

    monkeypatch.setattr(context.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="failed with code 1"):
        context.run_meta_text(
            config={"meta_evolver": {"backend": "codex", "codex": {"command": "fake"}}},
            project_root=tmp_path,
            output_dir=tmp_path / "stage",
            inputs={},
            prompt="Return a result",
        )
    assert len(calls) == 1
