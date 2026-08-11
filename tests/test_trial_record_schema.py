"""Validation and integrity tests for canonical trial records."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "trial-record.schema.json"
EXAMPLE_PATHS = {
    "valid": PROJECT_ROOT / "schemas" / "examples" / "trial-record.valid.json",
    "invalid_response": PROJECT_ROOT / "schemas" / "examples" / "trial-record.invalid-response.json",
    "provider_error": PROJECT_ROOT / "schemas" / "examples" / "trial-record.provider-error.json",
}


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def schema():
    return load_json(SCHEMA_PATH)


@pytest.fixture(scope="module")
def examples():
    return {name: load_json(path) for name, path in EXAMPLE_PATHS.items()}


@pytest.fixture(scope="module")
def validator(schema):
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def errors_for(validator, instance):
    return sorted(validator.iter_errors(instance), key=lambda error: list(error.path))


def test_all_trial_state_examples_validate(validator, examples):
    for state, example in examples.items():
        errors = errors_for(validator, example)
        assert errors == [], f"{state} failed with {[error.message for error in errors]}"


def test_example_text_hashes_match(examples):
    for example in examples.values():
        prompt = example["prompt"]
        assert hashlib.sha256(prompt["text"].encode()).hexdigest() == prompt["sha256"]

        response = example["response"]
        if response["raw_text"] is None:
            assert response["sha256"] is None
        else:
            assert hashlib.sha256(response["raw_text"].encode()).hexdigest() == response[
                "sha256"
            ]


@pytest.mark.parametrize(
    "example_name, mutation",
    [
        ("valid", lambda item: item["response"].update(raw_text=None, sha256=None)),
        ("valid", lambda item: item["parser"].update(status="rejected")),
        ("valid", lambda item: item["parser"].update(parsed_value=None)),
        ("valid", lambda item: item["validity"].update(reason="unexpected")),
        ("valid", lambda item: item.update(error={"message": "unexpected"})),
        ("invalid_response", lambda item: item["parser"].update(status="parsed")),
        ("invalid_response", lambda item: item["parser"].update(error_code=None)),
        ("invalid_response", lambda item: item["trial_metrics"].update(value=0)),
        ("provider_error", lambda item: item["response"].update(raw_text="error body")),
        ("provider_error", lambda item: item.update(error=None)),
        ("provider_error", lambda item: item["transport"].update(attempts=4)),
        ("valid", lambda item: item.update(completed_at="2026-08-11T10:00:01-04:00")),
        ("valid", lambda item: item.update(unexpected=True)),
    ],
)
def test_inconsistent_trial_states_are_rejected(
    validator, examples, example_name, mutation
):
    instance = copy.deepcopy(examples[example_name])
    mutation(instance)
    errors = errors_for(validator, instance)
    assert errors, "Inconsistent trial state unexpectedly passed validation"


def test_interrupted_trial_requires_error_and_empty_metrics(validator, examples):
    instance = copy.deepcopy(examples["provider_error"])
    instance["validity"].update(
        status="interrupted",
        reason_code="process_interrupted",
        reason="The local process stopped before the trial completed",
    )
    instance["error"].update(
        category="internal",
        code="keyboard_interrupt",
        message="The runner was interrupted",
        retryable=False,
    )
    assert errors_for(validator, instance) == []

    instance["trial_metrics"]["choice"] = "A"
    assert errors_for(validator, instance)
