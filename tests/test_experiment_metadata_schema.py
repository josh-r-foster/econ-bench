"""Validation tests for canonical experiment metadata."""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "experiment-metadata.schema.json"
EXAMPLE_PATH = PROJECT_ROOT / "schemas" / "examples" / "experiment-metadata.json"
EXPERIMENTS_PATH = PROJECT_ROOT / "config" / "experiments.json"
MODELS_PATH = PROJECT_ROOT / "config" / "models.json"


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def schema():
    return load_json(SCHEMA_PATH)


@pytest.fixture(scope="module")
def example():
    return load_json(EXAMPLE_PATH)


@pytest.fixture(scope="module")
def validator(schema):
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validation_messages(validator, instance):
    return [error.message for error in validator.iter_errors(instance)]


def test_example_validates(validator, example):
    assert validation_messages(validator, example) == []


def test_schema_versions_match_manifests(schema):
    experiments = load_json(EXPERIMENTS_PATH)
    models = load_json(MODELS_PATH)
    assert schema["properties"]["benchmark_version"]["const"] == experiments[
        "benchmark_version"
    ] == models["benchmark_version"]
    assert schema["properties"]["schema_version"]["const"] == experiments[
        "schema_version"
    ] == models["schema_version"]


def test_example_matches_model_and_experiment_manifests(example):
    experiment_manifest = load_json(EXPERIMENTS_PATH)
    model_manifest = load_json(MODELS_PATH)
    experiment = next(
        item for item in experiment_manifest["experiments"]
        if item["id"] == example["experiment"]["id"]
    )
    model = next(
        item for item in model_manifest["models"]
        if item["id"] == example["model"]["id"]
    )

    assert example["experiment"]["family"] == experiment["family"]
    assert example["experiment"]["parameters"] == experiment["settings"]
    assert example["experiment"]["manifest_version"] == experiment_manifest[
        "manifest_version"
    ]
    assert example["model"]["provider"] == model["provider"]
    assert example["model"]["api_model_id"] == model["api_model_id"]
    assert example["model"]["parameters"]["requested_temperature"] == experiment[
        "temperature"
    ]
    assert example["model"]["parameters"]["max_output_tokens"] == experiment[
        "max_output_tokens"
    ]

    shared = experiment_manifest["shared_settings"]
    assert example["protocol"]["condition_order"] == shared["condition_order"]
    assert example["protocol"]["local_random_seed"] == shared["local_random_seed"]
    assert example["protocol"]["transport_retry_policy"] == shared[
        "transport_retry_policy"
    ]
    assert example["protocol"]["invalid_response_policy"] == shared[
        "invalid_response_policy"
    ]


def test_metadata_can_represent_every_active_experiment(validator, example):
    manifest = load_json(EXPERIMENTS_PATH)
    for experiment in manifest["experiments"]:
        instance = copy.deepcopy(example)
        instance["experiment"].update(
            id=experiment["id"],
            family=experiment["family"],
            parameters=experiment["settings"],
        )
        instance["model"]["parameters"]["requested_temperature"] = experiment[
            "temperature"
        ]
        instance["model"]["parameters"]["effective_temperature"] = experiment[
            "temperature"
        ]
        instance["model"]["parameters"]["max_output_tokens"] = experiment[
            "max_output_tokens"
        ]
        parser_names = experiment.get(
            "response_parsers", [experiment.get("response_parser")]
        )
        instance["protocol"]["response_parsers"] = parser_names
        assert validation_messages(validator, instance) == [], experiment["id"]


@pytest.mark.parametrize(
    "mutation, expected_fragment",
    [
        (lambda item: item.pop("schema_version"), "schema_version"),
        (lambda item: item.update(schema_version="2.0.0"), "1.0.0"),
        (lambda item: item["run"].update(started_at="2026-08-11T14:00:00-04:00"), "Z$"),
        (lambda item: item["run"].update(completed_at=None), "string"),
        (lambda item: item["provenance"].update(code_revision=None), "string"),
        (lambda item: item["model"].update(unexpected=True), "not allowed"),
    ],
)
def test_invalid_metadata_fails_clearly(
    validator, example, mutation, expected_fragment
):
    instance = copy.deepcopy(example)
    mutation(instance)
    messages = validation_messages(validator, instance)
    assert messages, "Invalid metadata unexpectedly passed validation"
    assert any(expected_fragment in message for message in messages), messages


def test_incomplete_migration_names_missing_fields(validator, example):
    instance = copy.deepcopy(example)
    instance["run"]["started_at"] = None
    instance["provenance"].update(
        capture_method="legacy_migration",
        completeness="incomplete",
        code_revision=None,
        repository_dirty=None,
        runner=None,
        python_version=None,
        platform=None,
        missing_fields=[
            "run.started_at",
            "provenance.code_revision",
            "provenance.repository_dirty",
            "provenance.runner",
            "provenance.python_version",
            "provenance.platform",
        ],
        source_paths=["data/results/social_preferences/gpt-4o/results.json"],
    )
    assert validation_messages(validator, instance) == []

    instance["provenance"]["missing_fields"] = []
    messages = validation_messages(validator, instance)
    assert any("non-empty" in message for message in messages), messages
