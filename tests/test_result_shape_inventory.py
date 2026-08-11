"""Tests for the result shape inventory scanner."""

import importlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def scanner():
    return importlib.import_module("scripts.inventory_result_shapes")


def test_shape_is_stable_across_object_key_order():
    module = scanner()
    first = {"trial": {"valid": True, "value": 3}, "metrics": [1.0, 2.0]}
    second = {"metrics": [3.0], "trial": {"value": 7, "valid": False}}
    assert module.shape_id(module.shape_of(first)) == module.shape_id(module.shape_of(second))


def test_inventory_reports_json_and_jsonl_variants(tmp_path):
    module = scanner()
    raw_path = tmp_path / "data" / "results" / "dictator" / "model" / "results.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(json.dumps({"trials": [{"value": 4}]}), encoding="utf-8")

    log_path = tmp_path / "data" / "results" / "runs" / "session.jsonl"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        json.dumps({"event": "model_call", "valid": True}) + "\n",
        encoding="utf-8",
    )

    report = module.inventory([tmp_path], project_root=tmp_path)
    assert report["summary"] == {
        "files_discovered": 2,
        "files_parsed": 2,
        "json_files": 1,
        "jsonl_files": 1,
        "jsonl_records": 1,
        "shape_variants": 2,
        "invalid_files": 0,
        "unclassified_files": 0,
    }
    assert {variant["family"] for variant in report["variants"]} == {
        "dictator_raw",
        "trace_log",
    }


def test_every_committed_dashboard_json_is_classified_and_valid():
    module = scanner()
    dashboard_path = PROJECT_ROOT / "web" / "data"
    report = module.inventory([dashboard_path])
    expected_files = len(list(dashboard_path.glob("*.json")))

    assert report["summary"]["files_parsed"] == expected_files
    assert report["summary"]["invalid_files"] == 0
    assert all(record["family"] != "unclassified" for record in report["files"])
