"""Validation tests for experiment-specific metric contracts."""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "experiment-metrics.schema.json"
EXAMPLES_PATH = PROJECT_ROOT / "schemas" / "examples" / "experiment-metrics.json"
TRIAL_EXAMPLE_PATH = PROJECT_ROOT / "schemas" / "examples" / "trial-record.valid.json"
EXPERIMENTS_PATH = PROJECT_ROOT / "config" / "experiments.json"


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def schema():
    return load_json(SCHEMA_PATH)


@pytest.fixture(scope="module")
def examples():
    return load_json(EXAMPLES_PATH)


@pytest.fixture(scope="module")
def validator(schema):
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def errors_for(validator, instance):
    return sorted(validator.iter_errors(instance), key=lambda error: list(error.path))


def all_examples(examples):
    for level_examples in examples.values():
        yield from level_examples.items()


def test_trial_and_aggregate_examples_cover_active_experiments(schema, examples):
    manifest = load_json(EXPERIMENTS_PATH)
    active = {
        experiment["id"]
        for experiment in manifest["experiments"]
        if experiment["status"] == "active"
    }
    schema_ids = set(schema["properties"]["experiment_id"]["enum"])

    assert set(examples["trial_examples"]) == active
    assert set(examples["aggregate_examples"]) == active
    assert schema_ids == active


def test_all_metric_examples_validate(validator, examples):
    for experiment_id, example in all_examples(examples):
        errors = errors_for(validator, example)
        assert errors == [], (
            f"{experiment_id} {example['metric_level']} failed with "
            f"{[error.message for error in errors]}"
        )


@pytest.mark.parametrize("level_key", ["trial_examples", "aggregate_examples"])
def test_every_metric_object_rejects_unknown_fields(validator, examples, level_key):
    for experiment_id, example in examples[level_key].items():
        instance = copy.deepcopy(example)
        instance["metrics"]["unexpected"] = 1
        assert errors_for(validator, instance), experiment_id


@pytest.mark.parametrize("level_key", ["trial_examples", "aggregate_examples"])
def test_every_metric_object_requires_its_fields(validator, examples, level_key):
    for experiment_id, example in examples[level_key].items():
        instance = copy.deepcopy(example)
        first_key = next(iter(instance["metrics"]))
        del instance["metrics"][first_key]
        assert errors_for(validator, instance), experiment_id


def test_dispatch_rejects_metrics_from_another_experiment(validator, examples):
    trial_ids = list(examples["trial_examples"])
    for index, experiment_id in enumerate(trial_ids):
        other_id = trial_ids[(index + 1) % len(trial_ids)]
        instance = copy.deepcopy(examples["trial_examples"][experiment_id])
        instance["metrics"] = copy.deepcopy(
            examples["trial_examples"][other_id]["metrics"]
        )
        assert errors_for(validator, instance), f"{experiment_id} accepted {other_id}"


@pytest.mark.parametrize(
    "experiment_id, mutation",
    [
        (
            "dictator",
            lambda item: item["metrics"].update(transfer_share=1.01),
        ),
        (
            "ultimatum",
            lambda item: item["metrics"].update(role="observer"),
        ),
        (
            "trust_game",
            lambda item: item["metrics"].update(return_share_of_received=-0.01),
        ),
        (
            "beauty_contest",
            lambda item: item["metrics"].update(guess=101),
        ),
        (
            "travellers_dilemma",
            lambda item: item["metrics"].update(claim_on_2_100_scale=1),
        ),
        (
            "matching_pennies",
            lambda item: item["metrics"].update(choice="edges"),
        ),
    ],
)
def test_trial_metric_bounds_and_categories_are_enforced(
    validator, examples, experiment_id, mutation
):
    instance = copy.deepcopy(examples["trial_examples"][experiment_id])
    mutation(instance)
    assert errors_for(validator, instance)


def test_aggregate_rates_use_unit_interval_shares(validator, examples):
    instance = copy.deepcopy(examples["aggregate_examples"]["public_goods"])
    instance["metrics"]["overall_mean_contribution_share"] = 50
    assert errors_for(validator, instance)


def test_empty_partial_aggregate_can_report_null_metrics(validator, examples):
    instance = copy.deepcopy(examples["aggregate_examples"]["dictator"])
    instance["metrics"]["sample"] = {
        "observed_trials": 0,
        "valid_trials": 0,
        "invalid_response_trials": 0,
        "provider_error_trials": 0,
        "interrupted_trials": 0,
        "valid_rate": None,
        "invalid_response_rate": None,
    }
    instance["metrics"]["overall_mean_transfer_share"] = None
    instance["metrics"]["by_pool"] = []
    assert errors_for(validator, instance) == []


def test_canonical_trial_metrics_validate_with_experiment_context(validator):
    trial = load_json(TRIAL_EXAMPLE_PATH)
    metric_object = {
        "experiment_id": "dictator",
        "metric_level": "trial",
        "metrics": trial["trial_metrics"],
    }
    assert errors_for(validator, metric_object) == []


def test_invalid_trial_empty_metrics_are_not_substantive_metric_objects(validator):
    metric_object = {
        "experiment_id": "dictator",
        "metric_level": "trial",
        "metrics": {},
    }
    assert errors_for(validator, metric_object)


def test_root_discriminator_and_nested_condition_are_closed(validator, examples):
    instance = copy.deepcopy(examples["aggregate_examples"]["matching_pennies"])
    instance["unexpected"] = True
    assert errors_for(validator, instance)

    instance = copy.deepcopy(examples["aggregate_examples"]["matching_pennies"])
    instance["metrics"]["by_win_payoff"][0]["unexpected"] = True
    assert errors_for(validator, instance)
