"""Schema and application validation for canonical results."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from .aggregation import aggregate_trials
from .provenance import normalize_timestamp, text_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ValidationFinding:
    code: str
    message: str
    location: str = ""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _validators() -> tuple[Draft202012Validator, Draft202012Validator]:
    schema_dir = PROJECT_ROOT / "schemas"
    result_schema = _load_json(schema_dir / "result-record.schema.json")
    metadata_schema = _load_json(schema_dir / "experiment-metadata.schema.json")
    trial_schema = _load_json(schema_dir / "trial-record.schema.json")
    metric_schema = _load_json(schema_dir / "experiment-metrics.schema.json")

    resources = [
        (schema["$id"], Resource.from_contents(schema))
        for schema in (metadata_schema, trial_schema)
    ]
    registry = Registry().with_resources(resources)
    return (
        Draft202012Validator(
            result_schema,
            registry=registry,
            format_checker=FormatChecker(),
        ),
        Draft202012Validator(metric_schema),
    )


@lru_cache(maxsize=1)
def _manifest_indexes() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    models = _load_json(PROJECT_ROOT / "config" / "models.json")["models"]
    manifest = _load_json(PROJECT_ROOT / "config" / "experiments.json")
    experiments = manifest["experiments"]
    return (
        {model["id"]: model for model in models},
        {
            experiment["id"]: {
                **experiment,
                "manifest_version": manifest["manifest_version"],
            }
            for experiment in experiments
        },
    )


def _location(path: Iterable[Any]) -> str:
    return "/" + "/".join(str(part) for part in path)


def _schema_findings(record: dict[str, Any]) -> list[ValidationFinding]:
    result_validator, metric_validator = _validators()
    findings = [
        ValidationFinding("schema", error.message, _location(error.absolute_path))
        for error in result_validator.iter_errors(record)
    ]
    if findings:
        return findings

    experiment_id = record["metadata"]["experiment"]["id"]
    if record["record_type"] == "aggregate":
        metric_object = {
            "experiment_id": experiment_id,
            "metric_level": "aggregate",
            "metrics": record["aggregate_metrics"],
        }
        metric_prefix = "/aggregate_metrics"
    else:
        trial = record["trial"]
        if trial["validity"]["status"] != "valid":
            return findings
        metric_object = {
            "experiment_id": experiment_id,
            "metric_level": "trial",
            "metrics": trial["trial_metrics"],
        }
        metric_prefix = "/trial/trial_metrics"

    for error in metric_validator.iter_errors(metric_object):
        path = list(error.path)
        if path and path[0] == "metrics":
            path = path[1:]
        findings.append(
            ValidationFinding(
                "metric_schema",
                error.message,
                f"{metric_prefix}{_location(path) if path else ''}",
            )
        )
    return findings


def _metadata_findings(record: dict[str, Any]) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    metadata = record["metadata"]
    models, experiments = _manifest_indexes()
    model = metadata["model"]
    experiment = metadata["experiment"]
    run = metadata["run"]

    started_at = normalize_timestamp(run["started_at"])
    completed_at = (
        normalize_timestamp(run["completed_at"])
        if run["completed_at"] is not None
        else None
    )
    if completed_at is not None and started_at > completed_at:
        findings.append(
            ValidationFinding(
                "timestamp_order",
                "run start follows run completion",
                "/metadata/run",
            )
        )

    manifest_model = models.get(model["id"])
    if manifest_model is None:
        findings.append(
            ValidationFinding("unknown_model", model["id"], "/metadata/model/id")
        )
    elif (
        model["provider"] != manifest_model["provider"]
        or model["api_model_id"] != manifest_model["api_model_id"]
    ):
        findings.append(
            ValidationFinding(
                "model_manifest_mismatch",
                "provider identity does not match the model manifest",
                "/metadata/model",
            )
        )
    elif (
        metadata["provenance"]["capture_method"] == "native"
        and manifest_model["status"] != "active"
    ):
        findings.append(
            ValidationFinding(
                "inactive_native_model",
                "native collection requires an active model manifest entry",
                "/metadata/model/id",
            )
        )

    manifest_experiment = experiments.get(experiment["id"])
    if manifest_experiment is None:
        findings.append(
            ValidationFinding(
                "unknown_experiment", experiment["id"], "/metadata/experiment/id"
            )
        )
    elif (
        experiment["family"] != manifest_experiment["family"]
        or (
            metadata["provenance"]["capture_method"] == "native"
            and (
                experiment["manifest_version"]
                != manifest_experiment["manifest_version"]
                or experiment["parameters"] != manifest_experiment["settings"]
            )
        )
    ):
        findings.append(
            ValidationFinding(
                "experiment_manifest_mismatch",
                "experiment metadata does not match the manifest",
                "/metadata/experiment/family",
            )
        )

    provenance = metadata["provenance"]
    if (
        provenance["capture_method"] == "native"
        and provenance["repository_dirty"] is not False
    ):
        findings.append(
            ValidationFinding(
                "dirty_native_provenance",
                "native collection requires a clean repository snapshot",
                "/metadata/provenance/repository_dirty",
            )
        )

    return findings


def _trial_integrity_findings(record: dict[str, Any]) -> list[ValidationFinding]:
    if record["record_type"] != "trial":
        return []
    trial = record["trial"]
    findings: list[ValidationFinding] = []

    normalized_timestamps = {}
    for field in ("started_at", "completed_at"):
        try:
            normalized_timestamps[field] = normalize_timestamp(trial[field])
        except ValueError as error:
            findings.append(
                ValidationFinding("timestamp", str(error), f"/trial/{field}")
            )

    if (
        len(normalized_timestamps) == 2
        and normalized_timestamps["started_at"]
        > normalized_timestamps["completed_at"]
    ):
        findings.append(
            ValidationFinding(
                "timestamp_order",
                "trial start follows trial completion",
                "/trial",
            )
        )

    prompt = trial["prompt"]
    if text_sha256(prompt["text"]) != prompt["sha256"]:
        findings.append(
            ValidationFinding("prompt_digest", "prompt digest mismatch", "/trial/prompt")
        )

    response = trial["response"]
    expected_response_digest = (
        text_sha256(response["raw_text"]) if response["raw_text"] is not None else None
    )
    if response["sha256"] != expected_response_digest:
        findings.append(
            ValidationFinding(
                "response_digest", "response digest mismatch", "/trial/response"
            )
        )

    metric_role = trial["trial_metrics"].get("role")
    if metric_role is not None and metric_role != trial["role"]:
        findings.append(
            ValidationFinding(
                "role_mismatch",
                "metric role does not match trial role",
                "/trial/role",
            )
        )
    if trial["validity"]["status"] == "valid":
        findings.extend(_substantive_trial_findings(record))
    return findings


def _substantive_trial_findings(
    record: dict[str, Any],
) -> list[ValidationFinding]:
    """Check experiment-specific feasibility and metric identities."""
    trial = record["trial"]
    experiment_id = record["metadata"]["experiment"]["id"]
    condition = trial["condition"]
    metrics = trial["trial_metrics"]
    findings: list[ValidationFinding] = []

    def relation(ok: bool, message: str, location: str) -> None:
        if not ok:
            findings.append(
                ValidationFinding("substantive_relation", message, location)
            )

    def close(left: float, right: float) -> bool:
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)

    if experiment_id == "dictator":
        pool = condition["pool_amount"]
        amount = metrics["transfer_amount"]
        relation(0 <= amount <= pool, "transfer is outside the feasible set", "/trial/trial_metrics/transfer_amount")
        relation(close(metrics["transfer_share"], amount / pool), "transfer share does not match the transfer", "/trial/trial_metrics/transfer_share")
    elif experiment_id == "ultimatum":
        pool = condition["pool_amount"]
        if trial["role"] == "proposer":
            amount = metrics["offer_amount"]
            relation(0 <= amount <= pool, "offer is outside the feasible set", "/trial/trial_metrics/offer_amount")
            relation(close(metrics["offer_share"], amount / pool), "offer share does not match the offer", "/trial/trial_metrics/offer_share")
        else:
            amount = condition["offer_amount"]
            relation(0 <= amount <= pool, "presented offer is outside the feasible set", "/trial/condition/offer_amount")
            relation(close(condition["offer_share"], amount / pool), "presented offer share does not match the offer", "/trial/condition/offer_share")
    elif experiment_id == "trust_game":
        endowment = condition["endowment"]
        if trial["role"] == "sender":
            amount = metrics["amount_sent"]
            relation(0 <= amount <= endowment, "sender transfer is outside the feasible set", "/trial/trial_metrics/amount_sent")
            relation(close(metrics["send_share"], amount / endowment), "sender share does not match the transfer", "/trial/trial_metrics/send_share")
        else:
            sent = condition["sent_amount"]
            received = condition["received_amount"]
            amount = metrics["amount_returned"]
            relation(0 <= sent <= endowment, "presented sender transfer is outside the feasible set", "/trial/condition/sent_amount")
            relation(close(received, sent * condition["multiplier"]), "received amount does not match the multiplier", "/trial/condition/received_amount")
            relation(0 <= amount <= received, "receiver return is outside the feasible set", "/trial/trial_metrics/amount_returned")
            if received:
                relation(close(metrics["return_share_of_received"], amount / received), "receiver share does not match the return", "/trial/trial_metrics/return_share_of_received")
            else:
                relation(metrics["return_share_of_received"] is None, "zero receipts require a null return share", "/trial/trial_metrics/return_share_of_received")
            if sent:
                relation(close(metrics["return_multiple_of_sent"], amount / sent), "return multiple does not match the return", "/trial/trial_metrics/return_multiple_of_sent")
            else:
                relation(metrics["return_multiple_of_sent"] is None, "zero transfers require a null return multiple", "/trial/trial_metrics/return_multiple_of_sent")
    elif experiment_id == "stag_hunt":
        if "payoff_dominant_action_label" in condition:
            dominant = condition["payoff_dominant_action_label"]
            choice = trial["parser"]["parsed_value"]
            expected_action = "stag" if choice == dominant else "hare"
            relation(metrics["action"] == expected_action, "semantic action does not match the counterbalanced label", "/trial/trial_metrics/action")
            relation(metrics["payoff_dominant_choice"] == (expected_action == "stag"), "payoff dominance flag does not match the action", "/trial/trial_metrics/payoff_dominant_choice")
    elif experiment_id == "beauty_contest":
        guess = metrics["guess"]
        lower = condition.get("choice_lower_bound", 0)
        upper = condition.get("choice_upper_bound", 100)
        relation(lower <= guess <= upper, "guess is outside the feasible set", "/trial/trial_metrics/guess")
        relation(close(metrics["distance_from_nash"], abs(guess)), "distance does not match the guess", "/trial/trial_metrics/distance_from_nash")
    elif experiment_id == "centipede_game":
        relation(metrics["action"] in {"pass", "take"}, "action is not feasible", "/trial/trial_metrics/action")
        relation(metrics["backward_induction_consistent"] == (metrics["action"] == "take"), "backward induction flag does not match the action", "/trial/trial_metrics/backward_induction_consistent")
    elif experiment_id == "public_goods":
        endowment = condition["endowment"]
        amount = metrics["contribution_amount"]
        relation(0 <= amount <= endowment, "contribution is outside the feasible set", "/trial/trial_metrics/contribution_amount")
        relation(close(metrics["contribution_share"], amount / endowment), "contribution share does not match the contribution", "/trial/trial_metrics/contribution_share")
    elif experiment_id == "travellers_dilemma":
        low = condition["lower_bound"]
        high = condition["upper_bound"]
        claim = metrics["claim_amount"]
        increment = condition.get("claim_increment", 1)
        relation(low <= claim <= high, "claim is outside the feasible set", "/trial/trial_metrics/claim_amount")
        grid_steps = (claim - low) / increment
        relation(close(grid_steps, round(grid_steps)), "claim is outside the permitted grid", "/trial/trial_metrics/claim_amount")
        normalized = (claim - low) / (high - low)
        relation(close(metrics["normalized_claim"], normalized), "normalized claim does not match the claim", "/trial/trial_metrics/normalized_claim")
        relation(metrics["lower_bound_choice"] == close(claim, low), "lower bound flag does not match the claim", "/trial/trial_metrics/lower_bound_choice")
    elif experiment_id == "matching_pennies":
        relation(condition.get("payoff_role", "matching") in {"matching", "mismatching"}, "payoff role is not feasible", "/trial/condition/payoff_role")
        relation(metrics["choice"] in {"heads", "tails"}, "choice is not feasible", "/trial/trial_metrics/choice")
    elif experiment_id in {"independence", "time"}:
        relation(metrics["semantic_choice"] in ({"reference_lottery", "axis_lottery"} if experiment_id == "independence" else {"sooner", "later"}) or condition.get("phase", "").startswith("diagnostic"), "semantic choice is not feasible", "/trial/trial_metrics/semantic_choice")
        if "midpoint" in condition:
            relation(condition["lower_bound_before"] <= condition["midpoint"] <= condition["upper_bound_before"], "bisection midpoint is outside its bounds", "/trial/condition/midpoint")
    return findings


def _aggregate_findings(record: dict[str, Any]) -> list[ValidationFinding]:
    if record["record_type"] != "aggregate":
        return []
    metrics = record["aggregate_metrics"]
    sample = metrics.get("sample")
    if not isinstance(sample, dict):
        return []

    findings: list[ValidationFinding] = []
    state_total = sum(
        sample[name]
        for name in (
            "valid_trials",
            "invalid_response_trials",
            "provider_error_trials",
            "interrupted_trials",
        )
    )
    observed = sample["observed_trials"]
    if state_total != observed:
        findings.append(
            ValidationFinding(
                "sample_count_identity",
                "validity state counts do not sum to observed trials",
                "/aggregate_metrics/sample",
            )
        )

    expected_valid_rate = sample["valid_trials"] / observed if observed else None
    expected_invalid_rate = (
        sample["invalid_response_trials"] / observed if observed else None
    )
    for name, expected in (
        ("valid_rate", expected_valid_rate),
        ("invalid_response_rate", expected_invalid_rate),
    ):
        actual = sample[name]
        if (actual is None) != (expected is None):
            findings.append(
                ValidationFinding(
                    "sample_rate_identity",
                    f"{name} does not match its count",
                    f"/aggregate_metrics/sample/{name}",
                )
            )
        elif actual is not None and abs(actual - expected) > 1e-9:
            findings.append(
                ValidationFinding(
                    "sample_rate_identity",
                    f"{name} does not match its count",
                    f"/aggregate_metrics/sample/{name}",
                )
            )
    for rate_name, error_name in (
        ("valid_rate", "valid_rate_standard_error"),
        ("invalid_response_rate", "invalid_response_rate_standard_error"),
    ):
        if error_name not in sample:
            continue
        rate = sample[rate_name]
        expected = math.sqrt(rate * (1 - rate) / observed) if observed else None
        actual = sample[error_name]
        if (actual is None) != (expected is None) or (
            actual is not None and abs(actual - expected) > 1e-9
        ):
            findings.append(
                ValidationFinding(
                    "sample_uncertainty_identity",
                    f"{error_name} does not match its binomial estimator",
                    f"/aggregate_metrics/sample/{error_name}",
                )
            )
    return findings


def validate_record(record: dict[str, Any]) -> list[ValidationFinding]:
    """Validate one result record against schemas and application invariants."""
    schema_findings = _schema_findings(record)
    if schema_findings:
        return schema_findings
    return [
        *_metadata_findings(record),
        *_trial_integrity_findings(record),
        *_aggregate_findings(record),
    ]


def validate_trial_collection(
    records: list[dict[str, Any]],
) -> list[ValidationFinding]:
    """Validate cross-record invariants for one canonical raw file."""
    findings: list[ValidationFinding] = []
    if not records:
        return [ValidationFinding("empty_raw", "raw result file has no trials")]

    baseline_metadata = records[0].get("metadata")
    trial_ids: set[str] = set()
    sequence_indices: set[int] = set()
    for index, record in enumerate(records):
        prefix = f"/{index}"
        for finding in validate_record(record):
            findings.append(
                ValidationFinding(
                    finding.code, finding.message, f"{prefix}{finding.location}"
                )
            )
        if record.get("record_type") != "trial":
            findings.append(
                ValidationFinding("raw_record_type", "raw record is not a trial", prefix)
            )
            continue
        if record.get("metadata") != baseline_metadata:
            findings.append(
                ValidationFinding(
                    "mixed_metadata", "raw records do not share metadata", prefix
                )
            )
        trial = record["trial"]
        if trial["trial_id"] in trial_ids:
            findings.append(
                ValidationFinding(
                    "duplicate_trial_id", trial["trial_id"], f"{prefix}/trial/trial_id"
                )
            )
        trial_ids.add(trial["trial_id"])
        if trial["sequence_index"] in sequence_indices:
            findings.append(
                ValidationFinding(
                    "duplicate_sequence_index",
                    str(trial["sequence_index"]),
                    f"{prefix}/trial/sequence_index",
                )
            )
        sequence_indices.add(trial["sequence_index"])

    expected_indices = set(range(len(records)))
    if sequence_indices != expected_indices:
        findings.append(
            ValidationFinding(
                "sequence_index_gap",
                "sequence indices must cover zero through trial count minus one",
            )
        )
    findings.extend(_canonical_plan_findings(records))
    return findings


def _canonical_plan_findings(
    records: list[dict[str, Any]],
) -> list[ValidationFinding]:
    """Bind canonical runner output to the exact manifest-derived trial plan."""
    if not records:
        return []
    metadata = records[0].get("metadata", {})
    provenance = metadata.get("provenance", {})
    if provenance.get("capture_method") not in {"native", "fixture"}:
        return []
    if provenance.get("runner") not in {"scripts/run_benchmark.py", "fixture"}:
        return []

    from src.tasks.config import experiment_config
    from src.tasks.specs import (
        BISECTION_EXPERIMENTS,
        fixed_trial_plans,
        next_trial_plan,
    )

    experiment_id = metadata["experiment"]["id"]
    manifest_config = experiment_config(experiment_id)
    config = {
        **manifest_config,
        "settings": metadata["experiment"]["parameters"],
    }
    order_seed = metadata["protocol"]["order_seed"]
    fixed = (
        []
        if experiment_id in BISECTION_EXPERIMENTS
        else fixed_trial_plans(config, order_seed)
    )
    findings: list[ValidationFinding] = []
    prefix: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        trial = record["trial"]
        location = f"/{index}/trial"
        if trial["sequence_index"] != index:
            findings.append(
                ValidationFinding(
                    "sequence_order",
                    "raw record order does not match its sequence index",
                    f"{location}/sequence_index",
                )
            )
        try:
            plan = next_trial_plan(config, prefix, fixed, order_seed)
        except Exception as error:
            findings.append(
                ValidationFinding(
                    "canonical_plan_error",
                    f"could not reconstruct canonical plan with {error}",
                    location,
                )
            )
            return findings
        if plan is None:
            findings.append(
                ValidationFinding(
                    "unexpected_trial",
                    "trial is not present in the canonical plan",
                    location,
                )
            )
            prefix.append(record)
            continue

        comparisons = (
            ("trial_id", trial["trial_id"], plan.trial_id),
            ("condition_id", trial["condition_id"], plan.condition_id),
            ("condition", trial["condition"], plan.condition),
            ("repetition", trial["repetition"], plan.repetition),
            ("role", trial["role"], plan.role),
            ("prompt/text", trial["prompt"]["text"], plan.prompt),
            ("parser/name", trial["parser"]["name"], plan.parser_name),
        )
        for field, actual, expected in comparisons:
            if actual != expected:
                findings.append(
                    ValidationFinding(
                        "canonical_plan_mismatch",
                        f"{field} does not match the canonical plan",
                        f"{location}/{field}",
                    )
                )

        raw_response = trial["response"]["raw_text"]
        parsed = plan.parser(raw_response) if raw_response is not None else None
        status = trial["validity"]["status"]
        if status == "valid":
            if parsed is None:
                findings.append(
                    ValidationFinding(
                        "canonical_parse_mismatch",
                        "a valid response is rejected by the canonical parser",
                        f"{location}/response/raw_text",
                    )
                )
            else:
                if trial["parser"]["parsed_value"] != parsed.value:
                    findings.append(
                        ValidationFinding(
                            "canonical_parse_mismatch",
                            "parsed value does not reproduce from the raw response",
                            f"{location}/parser/parsed_value",
                        )
                    )
                if trial["trial_metrics"] != parsed.metrics:
                    findings.append(
                        ValidationFinding(
                            "canonical_metric_mismatch",
                            "trial metrics do not reproduce from the raw response",
                            f"{location}/trial_metrics",
                        )
                    )
        elif status == "invalid_response" and parsed is not None:
            findings.append(
                ValidationFinding(
                    "canonical_parse_mismatch",
                    "an invalid response is accepted by the canonical parser",
                    f"{location}/response/raw_text",
                )
            )
        prefix.append(record)

    if (
        metadata["run"]["status"] == "completed"
        and provenance.get("completeness") == "complete"
    ):
        try:
            remaining = next_trial_plan(config, prefix, fixed, order_seed)
        except Exception as error:
            findings.append(
                ValidationFinding(
                    "canonical_plan_error",
                    f"could not test complete coverage with {error}",
                )
            )
        else:
            if remaining is not None:
                findings.append(
                    ValidationFinding(
                        "incomplete_trial_plan",
                        f"completed run is missing canonical trial {remaining.trial_id}",
                    )
                )
    return findings


def validate_result_pair(
    raw_records: list[dict[str, Any]], derived_record: dict[str, Any]
) -> list[ValidationFinding]:
    """Validate metadata and sample agreement between raw and derived results."""
    findings = validate_trial_collection(raw_records)
    findings.extend(validate_record(derived_record))
    if not raw_records or derived_record.get("record_type") != "aggregate":
        return findings

    if raw_records[0]["metadata"] != derived_record["metadata"]:
        findings.append(
            ValidationFinding(
                "raw_derived_metadata",
                "raw and derived metadata differ",
            )
        )

    status_counts = {
        status: sum(
            record["trial"]["validity"]["status"] == status
            for record in raw_records
        )
        for status in (
            "valid",
            "invalid_response",
            "provider_error",
            "interrupted",
        )
    }
    sample = derived_record["aggregate_metrics"].get("sample", {})
    expected = {
        "observed_trials": len(raw_records),
        "valid_trials": status_counts["valid"],
        "invalid_response_trials": status_counts["invalid_response"],
        "provider_error_trials": status_counts["provider_error"],
        "interrupted_trials": status_counts["interrupted"],
    }
    for field, value in expected.items():
        if sample.get(field) != value:
            findings.append(
                ValidationFinding(
                    "raw_derived_sample",
                    f"{field} does not reproduce from raw trials",
                    f"/aggregate_metrics/sample/{field}",
                )
            )

    if any(finding.code in {"schema", "metric_schema"} for finding in findings):
        return findings
    try:
        reproduced = aggregate_trials(raw_records)
    except Exception as error:
        findings.append(
            ValidationFinding(
                "aggregation_error",
                f"canonical aggregation failed with {error}",
            )
        )
    else:
        if reproduced != derived_record["aggregate_metrics"]:
            findings.append(
                ValidationFinding(
                    "aggregate_reproduction",
                    "derived metrics do not reproduce from canonical raw trials",
                    "/aggregate_metrics",
                )
            )
    return findings
