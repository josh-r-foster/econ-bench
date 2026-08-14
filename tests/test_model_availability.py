"""Checks for the phase four provider availability review."""

import json
from pathlib import Path

from scripts import validate_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load(name):
    with (PROJECT_ROOT / "config" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def test_availability_review_matches_every_active_endpoint():
    models = load("models.json")["models"]
    availability = load("model_availability.json")
    active = {model["id"]: model for model in models if model["status"] == "active"}
    reviewed = {model["id"]: model for model in availability["models"]}

    assert set(reviewed) == set(active)
    assert availability["documented_unavailable_active_model_ids"] == []
    for model_id, record in reviewed.items():
        assert record["api_model_id"] == active[model_id]["api_model_id"]
        assert record["provider"] == active[model_id]["provider"]
        assert record["documentation_status"] == "documented_available"


def test_availability_review_records_near_term_anthropic_retirements():
    reviewed = {
        model["id"]: model for model in load("model_availability.json")["models"]
    }
    assert reviewed["claude-sonnet-4-5"]["earliest_retirement_date"] == "2026-09-29"
    assert reviewed["claude-haiku-4-5"]["earliest_retirement_date"] == "2026-10-15"
    assert reviewed["claude-opus-4-5"]["earliest_retirement_date"] == "2026-11-24"


def test_availability_review_records_provider_smoke_evidence():
    availability = load("model_availability.json")
    reviewed = {model["id"]: model for model in availability["models"]}
    verified = {
        model_id for model_id, record in reviewed.items()
        if record["account_access_status"] == "verified"
    }
    assert availability["live_provider_checks"] == "passed_for_each_active_provider"
    assert verified == {
        "gpt-4o-mini", "claude-haiku-4-5", "gemini-2.5-flash-lite"
    }
    assert set(availability["account_access_unverified_model_ids"]) == (
        set(reviewed) - verified
    )


def test_protocol_validator_includes_availability_review():
    assert validate_protocol.validate() == (16, 8, 264)
