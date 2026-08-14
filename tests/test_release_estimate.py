"""Deterministic checks for the phase four release estimate."""

import json
from pathlib import Path

from scripts.estimate_release import build_estimate


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_release_estimate_matches_frozen_trial_design():
    estimate = build_estimate()
    experiments = {item["id"]: item for item in estimate["experiments"]}
    assert experiments["independence"] == {
        **experiments["independence"],
        "primary_calls": 2376,
        "validation_calls": 216,
        "diagnostic_calls": 195,
        "total_calls": 2787,
    }
    assert experiments["time"] == {
        **experiments["time"],
        "primary_calls": 2592,
        "validation_calls": 252,
        "diagnostic_calls": 185,
        "total_calls": 3029,
    }
    assert estimate["calls_per_model"] == 8306
    assert estimate["release_calls"] == 132896


def test_release_estimate_prices_every_active_model():
    with (PROJECT_ROOT / "config" / "models.json").open(encoding="utf-8") as handle:
        active = {
            model["id"] for model in json.load(handle)["models"]
            if model["status"] == "active"
        }
    estimate = build_estimate()
    assert {row["id"] for row in estimate["model_costs"]} == active
    assert estimate["estimated_cost_low_usd"] > 0
    assert estimate["estimated_cost_high_usd"] > estimate["estimated_cost_low_usd"]


def test_release_estimate_makes_uncertainty_explicit():
    estimate = build_estimate()
    assert estimate["output_token_scenarios_per_call"] == {"low": 8, "high": 128}
    assert estimate["assumptions"]["retries_and_reruns_included"] is False
    assert estimate["estimated_serial_hours_high"] > estimate["estimated_serial_hours_low"]
