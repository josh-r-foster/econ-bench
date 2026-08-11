#!/usr/bin/env python3
"""Validate the canonical EconBench release matrix without provider calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.results.io import read_json, read_jsonl
from src.results.model_ids import model_id_to_path_component
from src.results.validation import validate_result_pair


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _cell_status(
    release_root: Path, model_id: str, experiment_id: str
) -> dict[str, Any]:
    model_key = model_id_to_path_component(model_id)
    derived_path = release_root / "derived" / model_key / f"{experiment_id}.json"
    if not derived_path.is_file():
        return {
            "status": "MISSING",
            "detail": str(derived_path.relative_to(release_root)),
        }

    try:
        derived = read_json(derived_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return {"status": "INVALID", "detail": f"derived parse error {error}"}

    try:
        metadata = derived["metadata"]
        run_id = metadata["run"]["id"]
    except (KeyError, TypeError) as error:
        return {"status": "INVALID", "detail": f"derived structure error {error}"}
    raw_path = release_root / "raw" / model_key / experiment_id / f"{run_id}.jsonl"
    if not raw_path.is_file():
        return {
            "status": "MISSING",
            "detail": str(raw_path.relative_to(release_root)),
        }

    try:
        raw_records = read_jsonl(raw_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return {"status": "INVALID", "detail": f"raw parse error {error}"}

    findings = validate_result_pair(raw_records, derived)
    if findings:
        first = findings[0]
        return {
            "status": "INVALID",
            "detail": f"{first.code} {first.location} {first.message}".strip(),
            "finding_count": len(findings),
        }

    provenance = metadata["provenance"]
    sample = derived["aggregate_metrics"]["sample"]
    if provenance["completeness"] != "complete":
        return {
            "status": "PARTIAL",
            "detail": "incomplete provenance",
            "valid_trials": sample["valid_trials"],
            "observed_trials": sample["observed_trials"],
        }
    if sample["invalid_response_rate"] is not None and sample["invalid_response_rate"] > 0.05:
        return {
            "status": "PARTIAL",
            "detail": "invalid response rate exceeds release threshold",
            "valid_trials": sample["valid_trials"],
            "observed_trials": sample["observed_trials"],
        }
    return {
        "status": "PASS",
        "detail": "canonical raw and derived results agree",
        "valid_trials": sample["valid_trials"],
        "observed_trials": sample["observed_trials"],
    }


def validate_release(
    release_root: Path,
    *,
    model_filter: set[str] | None = None,
) -> dict[str, Any]:
    """Validate every required active experiment cell and report exclusions."""
    models_manifest = _load(PROJECT_ROOT / "config" / "models.json")
    experiments_manifest = _load(PROJECT_ROOT / "config" / "experiments.json")
    release_matrix = _load(PROJECT_ROOT / "config" / "release_matrix.json")
    active_experiments = [
        item["id"]
        for item in experiments_manifest["experiments"]
        if item["status"] == "active"
    ]
    model_status = {item["id"]: item["status"] for item in models_manifest["models"]}
    matrix = release_matrix["matrix"]

    cells = []
    for model_id, experiment_cells in matrix.items():
        if model_filter is not None and model_id not in model_filter:
            continue
        for experiment_id in active_experiments:
            requirement = experiment_cells[experiment_id]
            if requirement == "excluded" or model_status[model_id] == "retired":
                result = {"status": "EXCLUDED", "detail": "matrix exclusion"}
            else:
                result = _cell_status(release_root, model_id, experiment_id)
            cells.append(
                {
                    "model_id": model_id,
                    "experiment_id": experiment_id,
                    "requirement": requirement,
                    **result,
                }
            )

    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell["status"]] = counts.get(cell["status"], 0) + 1
    return {
        "benchmark_version": models_manifest["benchmark_version"],
        "schema_version": models_manifest["schema_version"],
        "release_root": str(release_root),
        "active_experiments": active_experiments,
        "counts": counts,
        "cells": cells,
    }


def _print_report(report: dict[str, Any]) -> None:
    width = 31
    print(f"{'Model'.ljust(width)} {'Experiment'.ljust(24)} Status")
    print("-" * 72)
    for cell in report["cells"]:
        print(
            f"{cell['model_id'][:width - 1].ljust(width)} "
            f"{cell['experiment_id'].ljust(24)} {cell['status']}"
        )
    print("-" * 72)
    print(" ".join(f"{name}={count}" for name, count in sorted(report["counts"].items())))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "releases" / "1.0.0",
    )
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return success when cells are missing or partial but no cell is invalid",
    )
    args = parser.parse_args()
    report = validate_release(
        args.release_root,
        model_filter=set(args.models) if args.models else None,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)

    invalid = report["counts"].get("INVALID", 0)
    incomplete = report["counts"].get("MISSING", 0) + report["counts"].get("PARTIAL", 0)
    if invalid:
        return 1
    if incomplete and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
