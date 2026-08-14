"""End-to-end tests for canonical migration, aggregation, and projection."""

import copy
import json
from pathlib import Path

import pytest

from scripts.validate_results import _cell_status, validate_release
from src.results.aggregation import AGGREGATORS, aggregate_trials
from src.results.dashboard import (
    build_dashboard_projection,
    dashboard_filename,
    generate_dashboard_file,
)
from src.results.io import read_json, write_json, write_jsonl
from src.results.provenance import (
    format_utc_timestamp,
    normalize_code_revision,
    normalize_timestamp,
    text_sha256,
    utc_now,
)
from src.results.rationality import build_rationality_projection
from src.results.records import build_aggregate_result, build_trial, build_trial_result
from src.results.social_migration import migrate_legacy_social, write_social_migration
from src.results.validation import (
    validate_record,
    validate_result_pair,
    validate_trial_collection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def active_experiments():
    manifest = load_json(PROJECT_ROOT / "config" / "experiments.json")
    return [item for item in manifest["experiments"] if item["status"] == "active"]


def metadata_for(experiment):
    metadata = load_json(PROJECT_ROOT / "schemas" / "examples" / "experiment-metadata.json")
    metadata["experiment"].update(
        id=experiment["id"],
        family=experiment["family"],
        parameters=experiment["settings"],
    )
    return metadata


CONDITIONS = {
    "independence": {
        "axis": "y",
        "reference_p_low": 0.2,
        "reference_p_middle": 0.5,
        "reference_p_high": 0.3,
        "bisection_iteration": 1,
        "lower_bound_before": 0,
        "upper_bound_before": 1,
        "midpoint": 0.5,
    },
    "time": {
        "larger_amount": 100,
        "delay_days": 365,
        "front_end_delay_days": 0,
        "bisection_iteration": 1,
        "lower_bound_before": 0,
        "upper_bound_before": 100,
        "midpoint": 50,
    },
    "dictator": {"pool_amount": 100},
    "ultimatum": {"pool_amount": 100, "offer_amount": 20, "offer_share": 0.2},
    "trust_game": {
        "endowment": 100,
        "multiplier": 3,
        "sent_amount": 50,
        "sent_share": 0.5,
        "received_amount": 150,
    },
    "stag_hunt": {"coordination_payoff": 100, "safe_payoff_multiplier": 0.5},
    "beauty_contest": {"prize": 100},
    "centipede_game": {"final_payoff_level": 100, "turn": 1},
    "public_goods": {"endowment": 100, "multiplier": 1.5},
    "travellers_dilemma": {
        "upper_bound_level": 100,
        "lower_bound": 2,
        "upper_bound": 100,
        "bonus": 2,
    },
    "matching_pennies": {"win_payoff": 100, "lose_payoff": 0},
}


ROLES = {
    "ultimatum": "responder",
    "trust_game": "receiver",
}


def raw_fixture_for(experiment):
    metric_examples = load_json(
        PROJECT_ROOT / "schemas" / "examples" / "experiment-metrics.json"
    )["trial_examples"]
    experiment_id = experiment["id"]
    trial = build_trial(
        trial_id=f"{experiment_id}-fixture-r001",
        sequence_index=0,
        condition_id=f"{experiment_id}-fixture",
        condition=copy.deepcopy(CONDITIONS[experiment_id]),
        repetition=1,
        role=ROLES.get(experiment_id),
        started_at="2026-08-11T14:00:00Z",
        completed_at="2026-08-11T14:00:01Z",
        prompt_text=f"Fixture prompt for {experiment_id}",
        raw_response="fixture response",
        parser_name="parse_fixture",
        parser_status="parsed",
        parsed_value="fixture",
        validity_status="valid",
        trial_metrics=copy.deepcopy(metric_examples[experiment_id]["metrics"]),
    )
    return [build_trial_result(metadata_for(experiment), trial)]


def test_timestamp_and_revision_helpers_are_canonical():
    assert normalize_timestamp("2026-08-11T10:00:00-04:00") == (
        "2026-08-11T14:00:00.000000Z"
    )
    assert normalize_timestamp(
        "2026-08-11T10:00:00", "America/Toronto"
    ) == "2026-08-11T14:00:00.000000Z"
    with pytest.raises(ValueError, match="source_timezone"):
        normalize_timestamp("2026-08-11T10:00:00")
    with pytest.raises(ValueError):
        format_utc_timestamp(__import__("datetime").datetime(2026, 8, 11))

    assert utc_now().endswith("Z")
    assert normalize_code_revision("a" * 40) == "a" * 40
    with pytest.raises(ValueError):
        normalize_code_revision("A" * 40)


def test_trial_builder_preserves_interaction_and_hashes():
    experiment = next(item for item in active_experiments() if item["id"] == "dictator")
    record = raw_fixture_for(experiment)[0]
    trial = record["trial"]
    assert trial["prompt"]["text"] == "Fixture prompt for dictator"
    assert trial["prompt"]["sha256"] == text_sha256(trial["prompt"]["text"])
    assert trial["response"]["raw_text"] == "fixture response"
    assert trial["response"]["sha256"] == text_sha256("fixture response")
    assert trial["parser"]["parsed_value"] == "fixture"
    assert trial["validity"]["status"] == "valid"
    assert validate_record(record) == []


def test_relational_validator_rejects_an_impossible_transfer():
    experiment = next(
        item for item in active_experiments() if item["id"] == "dictator"
    )
    record = raw_fixture_for(experiment)[0]
    record["trial"]["trial_metrics"]["transfer_amount"] = 200
    record["trial"]["trial_metrics"]["transfer_share"] = 1
    assert "substantive_relation" in {
        finding.code for finding in validate_record(record)
    }


def test_aggregators_and_dashboard_projections_cover_every_active_experiment():
    experiments = active_experiments()
    assert set(AGGREGATORS) == {item["id"] for item in experiments}

    for experiment in experiments:
        raw_records = raw_fixture_for(experiment)
        metrics = aggregate_trials(raw_records)
        derived = build_aggregate_result(raw_records[0]["metadata"], metrics)
        assert validate_result_pair(raw_records, derived) == [], experiment["id"]
        projection = build_dashboard_projection(raw_records, derived)
        assert projection["benchmark_version"] == "1.0.0"
        assert projection["schema_version"] == "1.0.0"
        assert projection["model_id"] == "gpt-5.2"
        assert dashboard_filename(experiment["id"], "gpt-5.2").endswith(".json")
        json.dumps(projection)


def test_rationality_projection_consumes_canonical_aggregates():
    experiments = {item["id"]: item for item in active_experiments()}
    independence_raw = raw_fixture_for(experiments["independence"])
    time_raw = raw_fixture_for(experiments["time"])
    independence = build_aggregate_result(
        independence_raw[0]["metadata"], aggregate_trials(independence_raw)
    )
    time = build_aggregate_result(time_raw[0]["metadata"], aggregate_trials(time_raw))

    projection = build_rationality_projection(independence, time)
    assert projection["benchmark_version"] == "1.0.0"
    assert projection["schema_version"] == "1.0.0"
    assert projection["model"] == "gpt-5.2"
    assert projection["metrics"]["patience"]["discount_factor"] is None
    assert projection["metrics"]["risk"]["error_rate"] is None


def test_validation_detects_integrity_and_collection_failures():
    experiment = next(item for item in active_experiments() if item["id"] == "dictator")
    record = raw_fixture_for(experiment)[0]

    tampered = copy.deepcopy(record)
    tampered["trial"]["prompt"]["text"] = "changed"
    assert {finding.code for finding in validate_record(tampered)} == {"prompt_digest"}

    duplicate = [record, copy.deepcopy(record)]
    codes = {finding.code for finding in validate_trial_collection(duplicate)}
    assert "duplicate_trial_id" in codes
    assert "duplicate_sequence_index" in codes

    noncanonical_timestamp = copy.deepcopy(record)
    noncanonical_timestamp["trial"]["started_at"] = "2026-08-11T14:00:00Z"
    assert {finding.code for finding in validate_record(noncanonical_timestamp)} == {
        "schema"
    }

    derived = build_aggregate_result(record["metadata"], aggregate_trials([record]))
    derived["aggregate_metrics"]["overall_mean_transfer_share"] = 0.99
    assert {
        finding.code for finding in validate_result_pair([record], derived)
    } == {"aggregate_reproduction"}


@pytest.fixture
def social_migration():
    source = load_json(FIXTURE_DIR / "legacy_social.json")
    return source, migrate_legacy_social(
        source,
        source_path="tests/fixtures/legacy_social.json",
        source_timezone="America/Toronto",
        project_root=PROJECT_ROOT,
        code_revision="a" * 40,
        repository_dirty=False,
    )


def test_social_migration_is_split_valid_and_aggregate_preserving(social_migration):
    source, migrations = social_migration
    expected = load_json(FIXTURE_DIR / "canonical_social_summary.json")
    assert set(migrations) == {"dictator", "ultimatum"}

    for payload in migrations.values():
        assert validate_result_pair(payload["raw"], payload["derived"]) == []
        assert payload["derived"]["metadata"]["provenance"]["completeness"] == "incomplete"
        assert "trial.prompt.text" in payload["derived"]["metadata"]["provenance"][
            "missing_fields"
        ]

    dictator = migrations["dictator"]["derived"]["aggregate_metrics"]
    assert dictator["overall_mean_transfer_share"] == pytest.approx(
        expected["dictator"]["overall_mean_transfer_share"]
    )
    assert {
        f"{row['pool_amount']:g}": row["mean_transfer_share"]
        for row in dictator["by_pool"]
    } == pytest.approx(expected["dictator"]["mean_transfer_share_by_pool"])

    ultimatum = migrations["ultimatum"]["derived"]["aggregate_metrics"]
    assert ultimatum["overall_mean_offer_share"] == pytest.approx(
        expected["ultimatum"]["overall_mean_offer_share"]
    )
    assert {
        f"{row['pool_amount']:g}": row["minimum_acceptable_offer_share"]
        for row in ultimatum["responder_by_pool"]
    } == expected["ultimatum"]["minimum_acceptable_offer_share_by_pool"]

    raw_responses = [
        record["trial"]["response"]["raw_text"]
        for record in migrations["dictator"]["raw"]
    ]
    assert raw_responses == [trial["raw_response"] for trial in source["dictator_proposer"]]


def test_social_migration_never_counts_an_unverifiable_default():
    source = load_json(FIXTURE_DIR / "legacy_social.json")
    source["dictator_proposer"][0]["raw_response"] = "I refuse to provide an amount"
    migrations = migrate_legacy_social(
        source,
        source_path="tests/fixtures/legacy_social.json",
        source_timezone="America/Toronto",
        project_root=PROJECT_ROOT,
        code_revision="a" * 40,
        repository_dirty=False,
    )
    payload = migrations["dictator"]
    first = payload["raw"][0]["trial"]
    assert first["validity"]["status"] == "invalid_response"
    assert first["trial_metrics"] == {}
    sample = payload["derived"]["aggregate_metrics"]["sample"]
    assert sample["observed_trials"] == 4
    assert sample["valid_trials"] == 3
    assert sample["invalid_response_trials"] == 1
    assert validate_result_pair(payload["raw"], payload["derived"]) == []


def test_social_migration_rejects_a_missing_raw_response():
    source = load_json(FIXTURE_DIR / "legacy_social.json")
    source["dictator_proposer"][0]["raw_response"] = ""
    with pytest.raises(ValueError, match="nonempty raw response"):
        migrate_legacy_social(
            source,
            source_path="tests/fixtures/legacy_social.json",
            source_timezone="America/Toronto",
            project_root=PROJECT_ROOT,
            code_revision="a" * 40,
            repository_dirty=False,
        )


def test_social_migration_writes_canonical_files_and_regenerates_dashboard(
    tmp_path, social_migration
):
    source, migrations = social_migration
    release_root = tmp_path / "release"
    paths = write_social_migration(migrations, release_root)
    assert len(paths) == 4

    dashboard_dir = tmp_path / "dashboard"
    for experiment_id, payload in migrations.items():
        metadata = payload["derived"]["metadata"]
        run_id = metadata["run"]["id"]
        raw_path = release_root / "raw" / "gpt-4o" / experiment_id / f"{run_id}.jsonl"
        derived_path = release_root / "derived" / "gpt-4o" / f"{experiment_id}.json"
        generated = generate_dashboard_file(raw_path, derived_path, dashboard_dir)
        assert generated.is_file()

    dictator_projection = read_json(dashboard_dir / "dictator_experiment_gpt-4o.json")
    ultimatum_projection = read_json(dashboard_dir / "ultimatum_experiment_gpt-4o.json")
    assert [row["offer_percentage"] for row in dictator_projection["dictator_proposer"]] == [
        row["offer_percentage"] for row in source["dictator_proposer"]
    ]
    assert [row["decision"] for row in ultimatum_projection["ultimatum_responder"]] == [
        row["decision"] for row in source["ultimatum_responder"]
    ]


def test_release_validator_covers_all_active_experiments(tmp_path, social_migration):
    _, migrations = social_migration
    report = validate_release(tmp_path, model_filter={"gpt-5.2"})
    expected_ids = {item["id"] for item in active_experiments()}
    assert report["active_experiments"] == [item["id"] for item in active_experiments()]
    assert {cell["experiment_id"] for cell in report["cells"]} == expected_ids
    assert report["counts"] == {"MISSING": 11}

    write_social_migration(migrations, tmp_path)
    for experiment_id in migrations:
        assert _cell_status(tmp_path, "gpt-4o", experiment_id)["status"] == "PARTIAL"
