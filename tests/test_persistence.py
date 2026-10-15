"""Publication, interruption and append semantics for shared artifact writers."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.domains.evidence import write_records
from src.training.sampling import SamplingRecordWriter
from src.utils.persistence import append_jsonl, atomic_json, atomic_jsonl, atomic_text


def test_atomic_replacement_preserves_content_and_mode_on_failure(tmp_path):
    path = tmp_path / "state.json"
    atomic_json(path, {"incumbent": "old"})
    path.chmod(0o640)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        atomic_json(path, {"bad": float("nan")})
    assert path.read_bytes() == original
    with pytest.raises(KeyboardInterrupt), atomic_text(path) as stream:
        stream.write("partial")
        raise KeyboardInterrupt()
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))
    atomic_json(path, {"incumbent": "new"})
    assert json.loads(path.read_text()) == {"incumbent": "new"}
    assert path.stat().st_mode & 0o777 == 0o640


def test_jsonl_generator_failure_and_empty_evidence_keep_previous_artifact(tmp_path):
    path = tmp_path / "records.jsonl"
    atomic_jsonl(path, [{"id": "old"}])
    previous = path.read_bytes()

    def broken():
        yield {"id": "new"}
        raise RuntimeError("interrupted producer")

    with pytest.raises(RuntimeError, match="interrupted"):
        atomic_jsonl(path, broken())
    with pytest.raises(ValueError, match="empty"):
        write_records([], path)
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob(".*.tmp"))


def test_concurrent_replacements_publish_only_complete_documents(tmp_path):
    path = tmp_path / "state.json"

    def publish(i):
        atomic_json(path, {"generation": i, "values": [i] * 200})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(publish, range(40)))
    result = json.loads(path.read_text())
    assert result["values"] == [result["generation"]] * 200
    assert not list(tmp_path.glob(".*.tmp"))


def test_append_keeps_complete_records_after_error_and_resume(tmp_path):
    path = tmp_path / "progress.jsonl"
    with path.open("w") as stream:
        append_jsonl(stream, {"id": 1}, durable=True)
        with pytest.raises(TypeError):
            append_jsonl(stream, {"invalid": object()})
    with path.open("a") as stream:
        append_jsonl(stream, {"id": 2})
    assert [json.loads(line) for line in path.read_text().splitlines()] == [{"id": 1}, {"id": 2}]


def test_sampling_writer_retains_thread_lock_and_incremental_resume(tmp_path):
    accepted, attempts = tmp_path / "accepted.jsonl", tmp_path / "attempts.jsonl"
    with (
        SamplingRecordWriter(accepted, attempts, append=False) as writer,
        ThreadPoolExecutor(max_workers=8) as pool,
    ):
        list(pool.map(lambda i: writer.write_attempt({"sample_id": str(i)}), range(80)))
    with SamplingRecordWriter(accepted, attempts, append=True) as writer:
        writer.write_attempt({"sample_id": "resumed"})
    rows = [json.loads(line) for line in attempts.read_text().splitlines()]
    assert {r["sample_id"] for r in rows} == {*map(str, range(80)), "resumed"}
    assert len(rows) == 81
