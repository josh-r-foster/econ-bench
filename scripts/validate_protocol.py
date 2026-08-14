#!/usr/bin/env python3
"""Validate the frozen EconBench protocol manifests."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
sys.path.insert(0, str(ROOT))

from src.results.model_ids import model_id_to_path_component
from src.models.inference_controls import recorded_inference_controls


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
    availability_manifest = load_json(CONFIG_DIR / "model_availability.json")
    pricing_manifest = load_json(CONFIG_DIR / "model_pricing.json")
    dashboard_models = load_json(ROOT / "web" / "data" / "models.json")

    versions = {
        models_manifest.get("benchmark_version"),
        experiments_manifest.get("benchmark_version"),
        matrix_manifest.get("benchmark_version"),
        availability_manifest.get("benchmark_version"),
        pricing_manifest.get("benchmark_version"),
    }
    require(len(versions) == 1 and None not in versions, "Benchmark versions must agree")

    schema_versions = {
        models_manifest.get("schema_version"),
        experiments_manifest.get("schema_version"),
        matrix_manifest.get("schema_version"),
        availability_manifest.get("schema_version"),
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

    active_model_ids = {
        model_id for model_id, model in model_by_id.items()
        if model["status"] == "active"
    }
    for model_id in active_model_ids:
        model = model_by_id[model_id]
        controls = recorded_inference_controls(
            model["provider"], model["api_model_id"]
        )
        require(
            bool(controls["effective_reasoning_mode"]),
            f"Missing reasoning control for {model_id}",
        )
        require(
            bool(controls["provider_options"]),
            f"Missing provider options for {model_id}",
        )
        require(
            controls["provider_options"].get("sdk_max_retries") == 0,
            f"Provider SDK retries must be disabled for {model_id}",
        )
        require(
            bool(controls["provider_options"].get("sdk_version")),
            f"Provider SDK version must be recorded for {model_id}",
        )
    availability_records = availability_manifest.get("models")
    require(
        isinstance(availability_records, list) and availability_records,
        "The availability manifest must contain models",
    )
    availability_by_id: dict[str, dict[str, Any]] = {}
    provider_domains = {
        "openai": "developers.openai.com",
        "anthropic": "platform.claude.com",
        "google": "ai.google.dev",
    }
    for record in availability_records:
        model_id = record.get("id")
        require(model_id in active_model_ids, f"Availability review contains inactive model {model_id}")
        require(model_id not in availability_by_id, f"Duplicate availability record {model_id}")
        model = model_by_id[model_id]
        require(record.get("provider") == model["provider"], f"Availability provider mismatch for {model_id}")
        require(record.get("api_model_id") == model["api_model_id"], f"Availability endpoint mismatch for {model_id}")
        require(
            record.get("documentation_status") in {
                "documented_available", "documented_unavailable"
            },
            f"Invalid documentation status for {model_id}",
        )
        require(
            record.get("provider_lifecycle_status")
            in {"active", "available_snapshot", "stable"},
            f"Invalid active lifecycle status for {model_id}",
        )
        require(
            record.get("account_access_status") in {"verified", "unverified"},
            f"Invalid account access status for {model_id}",
        )
        source = record.get("source")
        require(
            isinstance(source, str)
            and urlparse(source).hostname == provider_domains[model["provider"]],
            f"Invalid availability source for {model_id}",
        )
        retirement = record.get("earliest_retirement_date")
        if retirement is not None:
            try:
                date.fromisoformat(retirement)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid retirement date for {model_id}") from error
        availability_by_id[model_id] = record
    require(
        set(availability_by_id) == active_model_ids,
        "The availability review must contain every active model exactly once",
    )
    documented_unavailable = {
        model_id for model_id, record in availability_by_id.items()
        if record["documentation_status"] == "documented_unavailable"
    }
    require(
        set(availability_manifest.get("documented_unavailable_active_model_ids", []))
        == documented_unavailable,
        "The documented unavailable model list is inconsistent",
    )
    require(
        not documented_unavailable,
        "A documented unavailable model must be retired before collection",
    )
    access_unverified = {
        model_id for model_id, record in availability_by_id.items()
        if record["account_access_status"] == "unverified"
    }
    require(
        set(availability_manifest.get("account_access_unverified_model_ids", []))
        == access_unverified,
        "The account access model list is inconsistent",
    )

    price_records = pricing_manifest.get("models")
    require(
        isinstance(price_records, list) and price_records,
        "The pricing manifest must contain models",
    )
    price_by_id: dict[str, dict[str, Any]] = {}
    for record in price_records:
        model_id = record.get("id")
        require(model_id in active_model_ids, f"Pricing contains inactive model {model_id}")
        require(model_id not in price_by_id, f"Duplicate price record {model_id}")
        require(
            isinstance(record.get("input"), (int, float)) and record["input"] > 0,
            f"Invalid input price for {model_id}",
        )
        require(
            isinstance(record.get("output"), (int, float)) and record["output"] > 0,
            f"Invalid output price for {model_id}",
        )
        source = record.get("source")
        provider = model_by_id[model_id]["provider"]
        require(
            isinstance(source, str)
            and urlparse(source).hostname == provider_domains[provider],
            f"Invalid pricing source for {model_id}",
        )
        price_by_id[model_id] = record
    require(
        set(price_by_id) == active_model_ids,
        "The pricing review must contain every active model exactly once",
    )

    experiments = experiments_manifest.get("experiments")
    require(isinstance(experiments, list) and experiments, "The experiment manifest must contain experiments")

    shared_settings = experiments_manifest["shared_settings"]
    canonical_locations = shared_settings["canonical_locations"]
    requested_temperature = shared_settings.get("requested_temperature")
    require(
        requested_temperature == 0.5,
        "The shared requested temperature must be 0.5",
    )
    require(
        shared_settings.get("condition_order")
        == "seeded_permutation_with_balanced_labels",
        "The condition order must use the seeded balanced policy",
    )
    require(
        shared_settings.get("reasoning_policy")
        == "lowest_supported_fixed_and_recorded",
        "The reasoning policy must be fixed and recorded",
    )
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
        require(
            experiment.get("temperature") == requested_temperature,
            f"Experiment {experiment_id} must use the shared requested temperature",
        )
        require(isinstance(experiment.get("max_output_tokens"), int), f"Missing output limit for {experiment_id}")
        require(bool(experiment.get("settings")), f"Missing settings for {experiment_id}")
        require((ROOT / experiment["script"]).is_file(), f"Missing script for {experiment_id}")
        require(
            bool(experiment.get("response_parser") or experiment.get("response_parsers")),
            f"Missing response parser for {experiment_id}",
        )
        experiment_by_id[experiment_id] = experiment

    matching = experiment_by_id["matching_pennies"]["settings"]
    require(
        matching.get("roles") == ["matching", "mismatching"],
        "Matching Pennies must contain both payoff roles",
    )
    require(
        matching.get("repetitions_per_condition", 0) >= 100,
        "Matching Pennies requires at least 100 repetitions per role and payoff",
    )

    for experiment_id in ("independence", "time"):
        settings = experiment_by_id[experiment_id]["settings"]
        responses = settings.get("responses_per_bisection_step")
        require(
            isinstance(responses, int) and responses >= 3 and responses % 2 == 1,
            f"{experiment_id} requires an odd repeated response count per midpoint",
        )
    require(
        experiment_by_id["independence"]["settings"].get(
            "quadratic_eu_beta_norm_tolerance"
        ) == 0.05,
        "Independence requires the frozen quadratic utility tolerance",
    )
    require(
        experiment_by_id["time"]["settings"].get(
            "present_bias_minimum_share_difference"
        ) == 0.02,
        "Time requires the frozen present bias threshold",
    )

    requirements = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    require(
        requirements and all("==" in requirement for requirement in requirements),
        "Every direct Python dependency must be pinned exactly",
    )

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
