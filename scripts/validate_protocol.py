#!/usr/bin/env python3
"""Validate the frozen EconBench protocol manifests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
sys.path.insert(0, str(ROOT))

from src.results.model_ids import model_id_to_path_component


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate() -> tuple[int, int, int]:
    models_manifest = load_json(CONFIG_DIR / "models.json")
    experiments_manifest = load_json(CONFIG_DIR / "experiments.json")
    matrix_manifest = load_json(CONFIG_DIR / "release_matrix.json")
    dashboard_models = load_json(ROOT / "web" / "data" / "models.json")

    versions = {
        models_manifest.get("benchmark_version"),
        experiments_manifest.get("benchmark_version"),
        matrix_manifest.get("benchmark_version"),
    }
    require(len(versions) == 1 and None not in versions, "Benchmark versions must agree")

    schema_versions = {
        models_manifest.get("schema_version"),
        experiments_manifest.get("schema_version"),
        matrix_manifest.get("schema_version"),
    }
    require(len(schema_versions) == 1 and None not in schema_versions, "Schema versions must agree")

    models = models_manifest.get("models")
    require(isinstance(models, list) and models, "The model manifest must contain models")

    model_by_id: dict[str, dict[str, Any]] = {}
    model_id_by_key: dict[str, str] = {}
    for model in models:
        model_id = model.get("id")
        require(isinstance(model_id, str) and model_id, "Every model needs an identifier")
        require(model_id not in model_by_id, f"Duplicate model identifier {model_id}")
        model_key = model_id_to_path_component(model_id)
        require(
            model_key not in model_id_by_key,
            f"Model path component collision for {model_id} and {model_id_by_key.get(model_key)}",
        )
        require(model.get("status") in {"active", "retired"}, f"Invalid model status for {model_id}")
        require(model.get("provider") in {"openai", "anthropic", "google"}, f"Invalid provider for {model_id}")
        require(bool(model.get("api_model_id")), f"Missing API model identifier for {model_id}")
        if model["status"] == "retired":
            require(bool(model.get("retired_reason")), f"Missing retirement reason for {model_id}")
        model_by_id[model_id] = model
        model_id_by_key[model_key] = model_id

    require(
        set(model_by_id) == set(dashboard_models),
        "The protocol must classify every dashboard model exactly once",
    )

    experiments = experiments_manifest.get("experiments")
    require(isinstance(experiments, list) and experiments, "The experiment manifest must contain experiments")

    canonical_locations = experiments_manifest["shared_settings"]["canonical_locations"]
    require(
        all("{model_key}" in canonical_locations[name] for name in ("raw", "derived")),
        "Canonical result paths must use model_key",
    )
    require(
        all("{model_id}" not in value for value in canonical_locations.values()),
        "Canonical result paths must not interpolate model_id",
    )

    experiment_by_id: dict[str, dict[str, Any]] = {}
    for experiment in experiments:
        experiment_id = experiment.get("id")
        require(isinstance(experiment_id, str) and experiment_id, "Every experiment needs an identifier")
        require(experiment_id not in experiment_by_id, f"Duplicate experiment identifier {experiment_id}")
        require(experiment.get("status") == "active", f"Manifest experiment {experiment_id} must be active")
        require(experiment.get("family") in {"elicitation", "strategic_game"}, f"Invalid family for {experiment_id}")
        require(isinstance(experiment.get("temperature"), (int, float)), f"Missing temperature for {experiment_id}")
        require(isinstance(experiment.get("max_output_tokens"), int), f"Missing output limit for {experiment_id}")
        require(bool(experiment.get("settings")), f"Missing settings for {experiment_id}")
        require((ROOT / experiment["script"]).is_file(), f"Missing script for {experiment_id}")
        require(
            bool(experiment.get("response_parser") or experiment.get("response_parsers")),
            f"Missing response parser for {experiment_id}",
        )
        experiment_by_id[experiment_id] = experiment

    excluded = experiments_manifest.get("excluded_tasks", [])
    excluded_by_id = {task.get("id"): task for task in excluded}
    require(set(excluded_by_id) == {"risk", "transitivity"}, "Risk and transitivity need explicit exclusions")
    for task_id, task in excluded_by_id.items():
        require(task.get("status") == "excluded", f"Invalid excluded task status for {task_id}")
        require(bool(task.get("reason")), f"Missing excluded task reason for {task_id}")
        require(bool(task.get("former_script")), f"Missing removed script record for {task_id}")
        require(
            not (ROOT / task["former_script"]).exists(),
            f"Excluded placeholder still exists for {task_id}",
        )

    shared = experiments_manifest.get("shared_settings", {})
    invalid_policy = shared.get("invalid_response_policy", {})
    require(invalid_policy.get("silent_imputation_allowed") is False, "Silent imputation must be disabled")
    require(invalid_policy.get("include_invalid_trials_in_metrics") is False, "Invalid trials must be excluded")
    require(isinstance(invalid_policy.get("maximum_experiment_invalid_rate"), (int, float)), "Missing invalid rate")
    require(isinstance(invalid_policy.get("minimum_condition_valid_rate"), (int, float)), "Missing condition rate")

    locations = shared.get("canonical_locations", {})
    for location_name in ("raw", "derived", "release_manifest", "dashboard_projection"):
        require(bool(locations.get(location_name)), f"Missing canonical location {location_name}")

    matrix = matrix_manifest.get("matrix")
    require(isinstance(matrix, dict), "The release matrix must be an object")
    require(set(matrix) == set(model_by_id), "The release matrix must contain every model")

    allowed_statuses = set(matrix_manifest.get("allowed_cell_statuses", []))
    require(allowed_statuses == {"required", "optional", "excluded"}, "Cell statuses are incomplete")

    experiment_ids = set(experiment_by_id)
    cell_count = 0
    for model_id, cells in matrix.items():
        require(set(cells) == experiment_ids, f"Incomplete experiment cells for {model_id}")
        require(set(cells.values()) <= allowed_statuses, f"Invalid cell status for {model_id}")
        expected = "required" if model_by_id[model_id]["status"] == "active" else "excluded"
        require(all(value == expected for value in cells.values()), f"Matrix policy mismatch for {model_id}")
        cell_count += len(cells)

    active_count = sum(model["status"] == "active" for model in models)
    retired_count = sum(model["status"] == "retired" for model in models)
    return active_count, retired_count, cell_count


def main() -> int:
    try:
        active_count, retired_count, cell_count = validate()
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Protocol validation failed. {error}")
        return 1

    print(
        "Protocol validation passed. "
        f"{active_count} active models. "
        f"{retired_count} retired models. "
        f"{cell_count} classified matrix cells."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
