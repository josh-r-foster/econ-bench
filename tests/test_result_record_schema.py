"""Composition and version tests for canonical result records."""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = PROJECT_ROOT / "schemas"
EXAMPLE_DIR = SCHEMA_DIR / "examples"
RESULT_SCHEMA_PATH = SCHEMA_DIR / "result-record.schema.json"
METADATA_SCHEMA_PATH = SCHEMA_DIR / "experiment-metadata.schema.json"
TRIAL_SCHEMA_PATH = SCHEMA_DIR / "trial-record.schema.json"
METRIC_SCHEMA_PATH = SCHEMA_DIR / "experiment-metrics.schema.json"
METADATA_EXAMPLE_PATH = EXAMPLE_DIR / "experiment-metadata.json"
TRIAL_EXAMPLE_PATH = EXAMPLE_DIR / "trial-record.valid.json"
METRIC_EXAMPLES_PATH = EXAMPLE_DIR / "experiment-metrics.json"
RESULT_EXAMPLE_PATHS = {
    "trial": EXAMPLE_DIR / "result-record.trial.json",
    "aggregate": EXAMPLE_DIR / "result-record.aggregate.json",
}
EXPERIMENTS_PATH = PROJECT_ROOT / "config" / "experiments.json"
MODELS_PATH = PROJECT_ROOT / "config" / "models.json"


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def schemas():
    return {
        "result": load_json(RESULT_SCHEMA_PATH),
        "metadata": load_json(METADATA_SCHEMA_PATH),
        "trial": load_json(TRIAL_SCHEMA_PATH),
        "metrics": load_json(METRIC_SCHEMA_PATH),
    }


@pytest.fixture(scope="module")
def examples():
    return {
        name: load_json(path) for name, path in RESULT_EXAMPLE_PATHS.items()
    }


@pytest.fixture(scope="module")
def result_validator(schemas):
    Draft202012Validator.check_schema(schemas["result"])
    resources = [
        (
            schemas[name]["$id"],
            Resource.from_contents(schemas[name]),
        )
        for name in ("metadata", "trial")
    ]
    registry = Registry().with_resources(resources)
    return Draft202012Validator(
        schemas["result"],
        registry=registry,
        format_checker=FormatChecker(),
    )


@pytest.fixture(scope="module")
def metric_validator(schemas):
    Draft202012Validator.check_schema(schemas["metrics"])
    return Draft202012Validator(schemas["metrics"])


def errors_for(validator, instance):
    return sorted(validator.iter_errors(instance), key=lambda error: list(error.path))


def test_trial_and_aggregate_result_examples_validate(result_validator, examples):
    for record_type, example in examples.items():
        errors = errors_for(result_validator, example)
        assert errors == [], (
            f"{record_type} failed with {[error.message for error in errors]}"
        )


def test_result_examples_reuse_canonical_components(examples):
    metadata = load_json(METADATA_EXAMPLE_PATH)
    trial = load_json(TRIAL_EXAMPLE_PATH)
    metric_examples = load_json(METRIC_EXAMPLES_PATH)

    assert examples["trial"]["metadata"] == metadata
    assert examples["aggregate"]["metadata"] == metadata
    assert examples["trial"]["trial"] == trial
    assert examples["aggregate"]["aggregate_metrics"] == metric_examples[
        "aggregate_examples"
    ]["dictator"]["metrics"]


def test_result_versions_match_both_manifests(schemas):
    metadata_properties = schemas["metadata"]["properties"]
    experiments = load_json(EXPERIMENTS_PATH)
    models = load_json(MODELS_PATH)

    assert metadata_properties["benchmark_version"]["const"] == experiments[
        "benchmark_version"
    ] == models["benchmark_version"]
    assert metadata_properties["schema_version"]["const"] == experiments[
        "schema_version"
    ] == models["schema_version"]


@pytest.mark.parametrize("record_type", ["trial", "aggregate"])
@pytest.mark.parametrize("version_field", ["benchmark_version", "schema_version"])
def test_every_result_requires_both_versions(
    result_validator, examples, record_type, version_field
):
    instance = copy.deepcopy(examples[record_type])
    del instance["metadata"][version_field]
    assert errors_for(result_validator, instance)


@pytest.mark.parametrize("record_type", ["trial", "aggregate"])
@pytest.mark.parametrize("version_field", ["benchmark_version", "schema_version"])
def test_every_result_rejects_unsupported_versions(
    result_validator, examples, record_type, version_field
):
    instance = copy.deepcopy(examples[record_type])
    instance["metadata"][version_field] = "2.0.0"
    assert errors_for(result_validator, instance)


@pytest.mark.parametrize(
    "record_type, mutation",
    [
        (
            "trial",
            lambda item: item.update(aggregate_metrics={"unexpected": True}),
        ),
        (
            "trial",
            lambda item: item.update(trial=None),
        ),
        (
            "aggregate",
            lambda item: item.update(trial={"unexpected": True}),
        ),
        (
            "aggregate",
            lambda item: item.update(aggregate_metrics=None),
        ),
        (
            "aggregate",
            lambda item: item.update(aggregate_metrics={}),
        ),
    ],
)
def test_result_type_controls_payload(
    result_validator, examples, record_type, mutation
):
    instance = copy.deepcopy(examples[record_type])
    mutation(instance)
    assert errors_for(result_validator, instance)


def test_result_envelope_rejects_unknown_fields(result_validator, examples):
    instance = copy.deepcopy(examples["trial"])
    instance["schema_version"] = "1.0.0"
    assert errors_for(result_validator, instance)


def test_result_schema_uses_versioned_component_identifiers(schemas):
    result_schema_text = json.dumps(schemas["result"])
    assert schemas["metadata"]["$id"] in result_schema_text
    assert schemas["trial"]["$id"] in result_schema_text


def test_composed_results_cover_all_active_experiments(
    result_validator, metric_validator, examples
):
    manifest = load_json(EXPERIMENTS_PATH)
    metric_examples = load_json(METRIC_EXAMPLES_PATH)

    for experiment in manifest["experiments"]:
        if experiment["status"] != "active":
            continue

        experiment_id = experiment["id"]

        trial_result = copy.deepcopy(examples["trial"])
        trial_result["metadata"]["experiment"].update(
            id=experiment_id,
            family=experiment["family"],
            parameters=experiment["settings"],
        )
        trial_result["trial"]["trial_metrics"] = copy.deepcopy(
            metric_examples["trial_examples"][experiment_id]["metrics"]
        )
        assert errors_for(result_validator, trial_result) == [], experiment_id
        assert errors_for(
            metric_validator,
            {
                "experiment_id": experiment_id,
                "metric_level": "trial",
                "metrics": trial_result["trial"]["trial_metrics"],
            },
        ) == [], experiment_id

        aggregate_result = copy.deepcopy(examples["aggregate"])
        aggregate_result["metadata"]["experiment"].update(
            id=experiment_id,
            family=experiment["family"],
            parameters=experiment["settings"],
        )
        aggregate_result["aggregate_metrics"] = copy.deepcopy(
            metric_examples["aggregate_examples"][experiment_id]["metrics"]
        )
        assert errors_for(result_validator, aggregate_result) == [], experiment_id
        assert errors_for(
            metric_validator,
            {
                "experiment_id": experiment_id,
                "metric_level": "aggregate",
                "metrics": aggregate_result["aggregate_metrics"],
            },
        ) == [], experiment_id
