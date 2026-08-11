"""Migration of legacy combined social results into canonical split records."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any

from .aggregation import aggregate_trials
from .io import write_json, write_jsonl
from .model_ids import model_id_to_path_component
from .provenance import git_provenance, normalize_timestamp
from .records import build_aggregate_result, build_trial, build_trial_result


MISSING_PROMPT = "Original prompt unavailable in the legacy source artifact."


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _parse_amount(response: str, pool: float) -> float | None:
    patterns = [
        (r"(\d+(?:\.\d+)?)\s*%", lambda value: pool * value / 100),
        (r"\$\s*(\d+(?:\.\d+)?)", lambda value: value),
        (r"(\d+(?:\.\d+)?)\s*dollars?", lambda value: value),
        (r"\b(\d+(?:\.\d+)?)\b", lambda value: value),
    ]
    for pattern, transform in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            value = transform(float(match.group(1)))
            return value if 0 <= value <= pool else None
    return None


def _parse_decision(response: str) -> str | None:
    upper = response.strip().upper()
    if upper.startswith("ACCEPT"):
        return "ACCEPT"
    if upper.startswith("REJECT"):
        return "REJECT"
    if re.search(r"\bACCEPT\b|\bYES\b", upper):
        return "ACCEPT"
    if re.search(r"\bREJECT\b|\bNO\b", upper):
        return "REJECT"
    return None


def _legacy_metadata(
    *,
    model_id: str,
    experiment_id: str,
    source_path: str,
    source_digest: str,
    started_at: str,
    completed_at: str,
    parameters: dict[str, Any],
    code_revision: str,
    repository_dirty: bool,
    project_root: Path,
) -> dict[str, Any]:
    models = _load_manifest(project_root / "config" / "models.json")["models"]
    experiments_manifest = _load_manifest(project_root / "config" / "experiments.json")
    model = next(item for item in models if item["id"] == model_id)
    experiment = next(
        item for item in experiments_manifest["experiments"] if item["id"] == experiment_id
    )
    run_id = f"legacy-{experiment_id}-{source_digest[:16]}"
    parser_names = experiment.get(
        "response_parsers", [experiment.get("response_parser")]
    )
    return {
        "benchmark_version": experiments_manifest["benchmark_version"],
        "schema_version": experiments_manifest["schema_version"],
        "experiment": {
            "id": experiment_id,
            "family": experiment["family"],
            "manifest_version": "0.0.0",
            "parameters": parameters,
        },
        "model": {
            "id": model_id,
            "provider": model["provider"],
            "api_model_id": model["api_model_id"],
            "parameters": {
                "requested_temperature": None,
                "effective_temperature": None,
                "max_output_tokens": experiment["max_output_tokens"],
                "requested_reasoning_mode": None,
                "effective_reasoning_mode": None,
                "seed": None,
                "system_prompt": None,
                "tools_enabled": False,
                "provider_options": {},
            },
        },
        "protocol": {
            "condition_order": "legacy_source_order",
            "local_random_seed": 0,
            "response_parsers": parser_names,
            "transport_retry_policy": {
                "maximum_retries": 0,
                "backoff_seconds": [],
                "retry_only_before_completion": True,
            },
            "invalid_response_policy": {
                "silent_imputation_allowed": True,
                "include_invalid_trials_in_metrics": True,
                "maximum_experiment_invalid_rate": 1,
                "minimum_condition_valid_rate": 0,
                "failed_bisection_step_invalidates_sequence": False,
                "failed_release_run_action": "legacy_not_applicable",
            },
        },
        "run": {
            "id": run_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "status": "completed",
            "attempt": 1,
        },
        "provenance": {
            "capture_method": "legacy_migration",
            "completeness": "incomplete",
            "code_revision": code_revision,
            "repository_dirty": repository_dirty,
            "runner": "scripts/migrate_legacy_social.py",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "missing_fields": [
                "model.parameters.effective_temperature",
                "model.parameters.requested_temperature",
                "protocol.original_settings",
                "provenance.original_code_revision",
                "trial.prompt.text",
                "trial.response.finish_reason",
                "trial.response.provider_request_id",
                "trial.transport.completion_tokens",
                "trial.transport.latency_ms",
                "trial.transport.prompt_tokens",
            ],
            "source_paths": [source_path],
        },
    }


def _amount_trial(
    *,
    source_trial: dict[str, Any],
    experiment_id: str,
    role: str | None,
    sequence_index: int,
    source_timezone: str,
) -> dict[str, Any]:
    pool = float(source_trial["pool_amount"])
    response = source_trial.get("raw_response")
    if not isinstance(response, str) or not response:
        raise ValueError("legacy trial is missing a nonempty raw response")
    parsed = _parse_amount(response, pool)
    stored = float(source_trial["offer_amount"])
    valid = parsed is not None and abs(parsed - stored) <= 1e-9
    timestamp = normalize_timestamp(source_trial["timestamp"], source_timezone)
    repetition = int(source_trial["trial_number"])
    condition_id = (
        f"{role}-pool-{pool:g}" if role is not None else f"pool-{pool:g}"
    )
    if experiment_id == "dictator":
        metrics = {
            "transfer_amount": stored,
            "transfer_share": stored / pool,
        }
    else:
        metrics = {
            "role": "proposer",
            "offer_amount": stored,
            "offer_share": stored / pool,
        }
    return build_trial(
        trial_id=f"{condition_id}-r{repetition:03d}",
        sequence_index=sequence_index,
        condition_id=condition_id,
        condition={"pool_amount": pool},
        repetition=repetition,
        role=role,
        started_at=timestamp,
        completed_at=timestamp,
        prompt_text=MISSING_PROMPT,
        raw_response=response,
        parser_name="parse_dollar_amount",
        parser_status="parsed" if valid else "rejected",
        parsed_value=parsed,
        validity_status="valid" if valid else "invalid_response",
        trial_metrics=metrics if valid else {},
        parser_error_code=None if valid else "legacy_parse_or_value_mismatch",
        parser_error_message=None if valid else "Raw response does not support stored amount",
        validity_reason_code=None if valid else "legacy_parse_or_value_mismatch",
        validity_reason=None if valid else "Stored amount cannot be verified from raw response",
    )


def _responder_trial(
    source_trial: dict[str, Any], sequence_index: int, source_timezone: str
) -> dict[str, Any]:
    pool = float(source_trial["pool_amount"])
    offer = float(source_trial["offer_amount"])
    offer_share = offer / pool
    response = source_trial.get("raw_response")
    if not isinstance(response, str) or not response:
        raise ValueError("legacy trial is missing a nonempty raw response")
    parsed = _parse_decision(response)
    stored = str(source_trial["decision"]).upper()
    valid = parsed is not None and parsed == stored
    timestamp = normalize_timestamp(source_trial["timestamp"], source_timezone)
    repetition = int(source_trial["trial_number"])
    condition_id = f"responder-pool-{pool:g}-offer-{offer_share:g}"
    return build_trial(
        trial_id=f"{condition_id}-r{repetition:03d}",
        sequence_index=sequence_index,
        condition_id=condition_id,
        condition={
            "pool_amount": pool,
            "offer_amount": offer,
            "offer_share": offer_share,
        },
        repetition=repetition,
        role="responder",
        started_at=timestamp,
        completed_at=timestamp,
        prompt_text=MISSING_PROMPT,
        raw_response=response,
        parser_name="parse_accept_reject",
        parser_status="parsed" if valid else "rejected",
        parsed_value=parsed,
        validity_status="valid" if valid else "invalid_response",
        trial_metrics={"role": "responder", "accepted": stored == "ACCEPT"}
        if valid
        else {},
        parser_error_code=None if valid else "legacy_parse_or_value_mismatch",
        parser_error_message=None if valid else "Raw response does not support stored decision",
        validity_reason_code=None if valid else "legacy_parse_or_value_mismatch",
        validity_reason=None if valid else "Stored decision cannot be verified from raw response",
    )


def migrate_legacy_social(
    source: dict[str, Any],
    *,
    source_path: str,
    source_timezone: str,
    project_root: str | Path,
    code_revision: str | None = None,
    repository_dirty: bool | None = None,
) -> dict[str, dict[str, Any]]:
    """Convert one combined social artifact into split canonical results."""
    root = Path(project_root)
    model_id = source["model_id"]
    source_digest = hashlib.sha256(
        json.dumps(source, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if code_revision is None or repository_dirty is None:
        detected_revision, detected_dirty = git_provenance(root)
        code_revision = code_revision or detected_revision
        repository_dirty = detected_dirty if repository_dirty is None else repository_dirty

    dictator_trials = [
        _amount_trial(
            source_trial=trial,
            experiment_id="dictator",
            role=None,
            sequence_index=index,
            source_timezone=source_timezone,
        )
        for index, trial in enumerate(source.get("dictator_proposer", []))
    ]
    ultimatum_trials = []
    for trial in source.get("ultimatum_proposer", []):
        ultimatum_trials.append(
            _amount_trial(
                source_trial=trial,
                experiment_id="ultimatum",
                role="proposer",
                sequence_index=len(ultimatum_trials),
                source_timezone=source_timezone,
            )
        )
    for trial in source.get("ultimatum_responder", []):
        ultimatum_trials.append(
            _responder_trial(trial, len(ultimatum_trials), source_timezone)
        )

    migrations = {}
    for experiment_id, trials in (
        ("dictator", dictator_trials),
        ("ultimatum", ultimatum_trials),
    ):
        if not trials:
            raise ValueError(f"legacy source has no {experiment_id} trials")
        timestamps = [trial["started_at"] for trial in trials] + [
            trial["completed_at"] for trial in trials
        ]
        pools = sorted({trial["condition"]["pool_amount"] for trial in trials})
        if experiment_id == "dictator":
            parameters = {
                "pool_amounts": pools,
                "repetitions_by_pool": {
                    f"{pool:g}": sum(
                        trial["condition"]["pool_amount"] == pool for trial in trials
                    )
                    for pool in pools
                },
            }
        else:
            parameters = {
                "pool_amounts": pools,
                "proposer_trials": sum(trial["role"] == "proposer" for trial in trials),
                "responder_trials": sum(trial["role"] == "responder" for trial in trials),
            }
        metadata = _legacy_metadata(
            model_id=model_id,
            experiment_id=experiment_id,
            source_path=source_path,
            source_digest=source_digest,
            started_at=min(timestamps),
            completed_at=max(timestamps),
            parameters=parameters,
            code_revision=code_revision,
            repository_dirty=bool(repository_dirty),
            project_root=root,
        )
        raw_records = [build_trial_result(metadata, trial) for trial in trials]
        aggregate_metrics = aggregate_trials(raw_records)
        migrations[experiment_id] = {
            "raw": raw_records,
            "derived": build_aggregate_result(metadata, aggregate_metrics),
        }
    return migrations


def write_social_migration(
    migrations: dict[str, dict[str, Any]],
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Write split migration outputs under one canonical release root."""
    root = Path(output_root)
    destinations = []
    for experiment_id, payload in migrations.items():
        metadata = payload["derived"]["metadata"]
        model_key = model_id_to_path_component(metadata["model"]["id"])
        run_id = metadata["run"]["id"]
        raw_path = root / "raw" / model_key / experiment_id / f"{run_id}.jsonl"
        derived_path = root / "derived" / model_key / f"{experiment_id}.json"
        destinations.append((payload, raw_path, derived_path))

    if not overwrite:
        for _, raw_path, derived_path in destinations:
            for path in (raw_path, derived_path):
                if path.exists():
                    raise FileExistsError(path)

    written: list[Path] = []
    for payload, raw_path, derived_path in destinations:
        write_jsonl(raw_path, payload["raw"])
        write_json(derived_path, payload["derived"])
        written.extend([raw_path, derived_path])
    return written
