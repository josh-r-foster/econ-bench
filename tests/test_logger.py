"""Tests for enabled and disabled JSONL model call logging."""

import importlib
import json


def fresh_logger(monkeypatch):
    logger = importlib.import_module("src.models.logger")
    monkeypatch.setattr(logger, "_initialized", False)
    monkeypatch.setattr(logger, "_log_path", None)
    return logger


def test_log_event_writes_one_json_line(monkeypatch, tmp_path):
    logger = fresh_logger(monkeypatch)
    monkeypatch.setenv("ECONBENCH_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("ECONBENCH_EXPERIMENT", "dictator")

    logger.log_event({"event": "fixture", "value": 3})

    paths = list(tmp_path.glob("session_*.jsonl"))
    assert len(paths) == 1
    records = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert records[0]["event"] == "fixture"
    assert records[0]["value"] == 3
    assert records[0]["experiment"] == "dictator"
    assert "timestamp" in records[0]


def test_model_call_records_dimensions_and_extra_fields(monkeypatch, tmp_path):
    logger = fresh_logger(monkeypatch)
    monkeypatch.setenv("ECONBENCH_LOG_DIR", str(tmp_path))
    logger.log_model_call(
        model="offline-fixture",
        prompt_chars=12,
        response="HEADS",
        latency_ms=4.26,
        prompt_tokens=3,
        completion_tokens=1,
        extra={"run_id": "run-001"},
    )

    path = next(tmp_path.glob("session_*.jsonl"))
    record = json.loads(path.read_text())
    assert record["response_chars"] == 5
    assert record["latency_ms"] == 4.3
    assert record["prompt_tokens"] == 3
    assert record["completion_tokens"] == 1
    assert record["run_id"] == "run-001"


def test_logging_can_be_disabled(monkeypatch, tmp_path):
    logger = fresh_logger(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ECONBENCH_LOG_DIR", "none")

    logger.log_event({"event": "fixture"})

    assert logger._log_path is None
    assert list(tmp_path.rglob("*.jsonl")) == []
