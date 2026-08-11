"""Canonical experiment execution with retries, checkpoints, and resume support."""

from __future__ import annotations

import argparse
import copy
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.results.aggregation import aggregate_trials
from src.results.io import read_json, read_jsonl, write_json, write_jsonl
from src.results.provenance import git_provenance, utc_now
from src.results.records import build_aggregate_result, build_trial, build_trial_result
from src.results.validation import validate_result_pair
from src.tasks.config import (
    PROJECT_ROOT,
    active_experiments,
    canonical_run_paths,
    experiment_config,
    experiment_manifest,
    model_config,
    validate_run_id,
)
from src.tasks.runtime import (
    ModelBinding,
    load_model_interface,
    request_model_completion,
)
from src.tasks.specs import (
    TrialPlan,
    bisection_conditions,
    bisection_plan,
    fixed_trial_plans,
)


BISECTION_EXPERIMENTS = {"independence", "time"}


class FixtureModel:
    """Deterministic offline provider used by the complete batch test."""

    model_id = "offline-fixture"

    def generate_response(self, *, prompt: str, **_kwargs):
        upper = prompt.upper()
        if 'ONLY "HEADS" OR "TAILS"' in upper:
            return "HEADS", None
        if 'ONLY "PASS" OR "TAKE"' in upper:
            return "TAKE", None
        if 'ONLY "ACCEPT" OR "REJECT"' in upper:
            return "ACCEPT", None
        if 'LETTER "A" OR "B"' in upper or 'ONLY "A" OR "B"' in upper:
            return "A", None
        if match := re.search(r"WHOLE NUMBER FROM (\d+) TO (\d+)", upper):
            return match.group(1), None
        if "WHOLE NUMBER FROM 0 TO 100" in upper:
            return "0", None
        if "JUST YOUR CHOSEN NUMBER" in upper:
            return "0", None
        return "$0", None


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")


def _metadata(
    model_id: str,
    config: dict[str, Any],
    run_id: str,
    started_at: str,
    *,
    runner: str,
    project_root: Path,
) -> dict[str, Any]:
    manifests = experiment_manifest()
    shared = manifests["shared_settings"]
    model = model_config(model_id)
    if model is None or model["status"] != "active":
        raise ValueError(f"canonical runs require an active registered model {model_id!r}")
    revision, dirty = git_provenance(project_root)
    parsers = config.get("response_parsers", [config.get("response_parser")])
    effective_temperature = config["temperature"]
    if model["provider"] == "openai" and model["api_model_id"].startswith(
        ("o1", "o3", "gpt-5")
    ):
        effective_temperature = None
    return {
        "benchmark_version": manifests["benchmark_version"],
        "schema_version": manifests["schema_version"],
        "experiment": {
            "id": config["id"],
            "family": config["family"],
            "manifest_version": manifests["manifest_version"],
            "parameters": copy.deepcopy(config["settings"]),
        },
        "model": {
            "id": model_id,
            "provider": model["provider"],
            "api_model_id": model["api_model_id"],
            "parameters": {
                "requested_temperature": config["temperature"],
                "effective_temperature": effective_temperature,
                "max_output_tokens": config["max_output_tokens"],
                "requested_reasoning_mode": None,
                "effective_reasoning_mode": None,
                "seed": shared["local_random_seed"],
                "system_prompt": shared["system_prompt"],
                "tools_enabled": shared["tools_enabled"],
                "provider_options": {},
            },
        },
        "protocol": {
            "condition_order": shared["condition_order"],
            "local_random_seed": shared["local_random_seed"],
            "response_parsers": parsers,
            "transport_retry_policy": copy.deepcopy(
                shared["transport_retry_policy"]
            ),
            "invalid_response_policy": copy.deepcopy(
                shared["invalid_response_policy"]
            ),
        },
        "run": {
            "id": run_id,
            "started_at": started_at,
            "completed_at": None,
            "status": "running",
            "attempt": 1,
        },
        "provenance": {
            "capture_method": "native",
            "completeness": "complete",
            "code_revision": revision,
            "repository_dirty": dirty,
            "runner": runner,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "missing_fields": [],
            "source_paths": [],
        },
    }


def _replace_metadata(records: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    for record in records:
        record["metadata"] = copy.deepcopy(metadata)


def _next_plan(
    config: dict[str, Any], records: list[dict[str, Any]], fixed: list[TrialPlan]
) -> TrialPlan | None:
    completed = {
        record["trial"]["trial_id"]
        for record in records
        if record["trial"]["validity"]["status"] != "interrupted"
    }
    if config["id"] not in BISECTION_EXPERIMENTS:
        return next((plan for plan in fixed if plan.trial_id not in completed), None)

    for base in bisection_conditions(config):
        trials = [
            record["trial"]
            for record in records
            if record["trial"]["condition_id"] == base["condition_id"]
            and record["trial"]["validity"]["status"] != "interrupted"
        ]
        plan = bisection_plan(config, base, trials)
        if plan is not None:
            return plan
    return None


def _trial_from_completion(plan: TrialPlan, sequence_index: int, completion) -> dict[str, Any]:
    if completion.status == "provider_error":
        error = {
            "category": "provider",
            "code": completion.error["type"] if completion.error else None,
            "message": completion.error["message"] if completion.error else "Provider failure",
            "retryable": True,
            "details": {"attempts": completion.attempts},
        }
        return build_trial(
            trial_id=plan.trial_id, sequence_index=sequence_index,
            condition_id=plan.condition_id, condition=plan.condition,
            repetition=plan.repetition, role=plan.role,
            started_at=completion.started_at, completed_at=completion.completed_at,
            prompt_text=plan.prompt, raw_response=None, parser_name=plan.parser_name,
            parser_status="not_run", parsed_value=None,
            validity_status="provider_error", trial_metrics={},
            validity_reason_code="provider_attempts_exhausted",
            validity_reason="Provider call failed after bounded retries",
            attempts=completion.attempts, latency_ms=completion.latency_ms, error=error,
        )

    response = completion.response if completion.response is not None else ""
    parsed = plan.parser(response) if response else None
    if parsed is None:
        return build_trial(
            trial_id=plan.trial_id, sequence_index=sequence_index,
            condition_id=plan.condition_id, condition=plan.condition,
            repetition=plan.repetition, role=plan.role,
            started_at=completion.started_at, completed_at=completion.completed_at,
            prompt_text=plan.prompt, raw_response=response,
            parser_name=plan.parser_name, parser_status="rejected", parsed_value=None,
            validity_status="invalid_response", trial_metrics={},
            parser_error_code="unparseable_response",
            parser_error_message="Response does not match the allowed response format",
            validity_reason_code="unparseable_response",
            validity_reason="Response does not identify a feasible substantive choice",
            attempts=completion.attempts, latency_ms=completion.latency_ms,
        )

    return build_trial(
        trial_id=plan.trial_id, sequence_index=sequence_index,
        condition_id=plan.condition_id, condition=plan.condition,
        repetition=plan.repetition, role=plan.role,
        started_at=completion.started_at, completed_at=completion.completed_at,
        prompt_text=plan.prompt, raw_response=response,
        parser_name=plan.parser_name, parser_status="parsed", parsed_value=parsed.value,
        validity_status="valid", trial_metrics=parsed.metrics,
        attempts=completion.attempts, latency_ms=completion.latency_ms,
    )


def _interrupted_trial(plan: TrialPlan, sequence_index: int, started_at: str) -> dict[str, Any]:
    completed_at = utc_now()
    return build_trial(
        trial_id=plan.trial_id, sequence_index=sequence_index,
        condition_id=plan.condition_id, condition=plan.condition,
        repetition=plan.repetition, role=plan.role,
        started_at=started_at, completed_at=completed_at,
        prompt_text=plan.prompt, raw_response=None, parser_name=plan.parser_name,
        parser_status="not_run", parsed_value=None, validity_status="interrupted",
        trial_metrics={}, validity_reason_code="keyboard_interrupt",
        validity_reason="Run was interrupted during the provider request",
        attempts=1, error={
            "category": "internal", "code": "keyboard_interrupt",
            "message": "Run was interrupted during the provider request",
            "retryable": True, "details": {},
        },
    )


def run_experiment(
    model_id: str,
    experiment_id: str,
    *,
    run_id: str,
    interface: Any | None = None,
    resume: bool = False,
    release_root: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
    verbose: bool = False,
    sleeper: Callable[[float], None] | None = None,
    runner: str = "scripts/run_benchmark.py",
) -> dict[str, Any]:
    """Run one canonical experiment and checkpoint every observed trial."""
    validate_run_id(run_id)
    root = Path(project_root)
    config = experiment_config(experiment_id)
    paths = canonical_run_paths(
        model_id, experiment_id, run_id, project_root=root, release_root=release_root
    )
    if paths.raw.exists() and not resume:
        raise FileExistsError(f"raw run already exists at {paths.raw}")

    records = read_jsonl(paths.raw) if paths.raw.exists() else []
    if records:
        metadata = copy.deepcopy(records[0]["metadata"])
        if metadata["run"]["id"] != run_id:
            raise ValueError("resume file has a different run identifier")
        if metadata["model"]["id"] != model_id:
            raise ValueError("resume file has a different model identifier")
        if metadata["experiment"]["id"] != experiment_id:
            raise ValueError("resume file has a different experiment identifier")
        if metadata["run"]["status"] == "completed" and paths.derived.is_file():
            derived = read_json(paths.derived)
            findings = validate_result_pair(records, derived)
            if findings:
                first = findings[0]
                raise ValueError(
                    f"completed resume input failed validation with {first.code} "
                    f"{first.message}"
                )
            return {"raw": records, "derived": derived, "paths": paths}
        metadata["run"]["status"] = "running"
        metadata["run"]["completed_at"] = None
        metadata["run"]["attempt"] += 1
        records = [
            record for record in records
            if record["trial"]["validity"]["status"] != "interrupted"
        ]
        for index, record in enumerate(records):
            record["trial"]["sequence_index"] = index
    else:
        metadata = _metadata(
            model_id, config, run_id, utc_now(), runner=runner, project_root=root
        )

    _replace_metadata(records, metadata)
    if records:
        write_jsonl(paths.raw, records)

    if interface is None:
        interface = load_model_interface(model_id)
    elif not hasattr(interface, "econbench_model_binding"):
        model = model_config(model_id)
        interface.econbench_model_binding = ModelBinding(
            model_id=model_id, api_model_id=model["api_model_id"],
            provider=model["provider"], status=model["status"], registered=True,
        )

    fixed = [] if experiment_id in BISECTION_EXPERIMENTS else fixed_trial_plans(config)
    retry = metadata["protocol"]["transport_retry_policy"]
    sleeper_function = sleeper if sleeper is not None else __import__("time").sleep

    while (plan := _next_plan(config, records, fixed)) is not None:
        started_at = utc_now()
        try:
            completion = request_model_completion(
                interface, experiment_id=experiment_id, prompt=plan.prompt,
                max_new_tokens=config["max_output_tokens"],
                temperature=config["temperature"], verbose=verbose,
                maximum_retries=retry["maximum_retries"],
                backoff_seconds=tuple(retry["backoff_seconds"]),
                sleeper=sleeper_function,
            )
            trial = _trial_from_completion(plan, len(records), completion)
        except KeyboardInterrupt:
            trial = _interrupted_trial(plan, len(records), started_at)
            records.append(build_trial_result(metadata, trial))
            metadata["run"]["status"] = "interrupted"
            metadata["run"]["completed_at"] = utc_now()
            _replace_metadata(records, metadata)
            write_jsonl(paths.raw, records)
            raise
        records.append(build_trial_result(metadata, trial))
        write_jsonl(paths.raw, records)

    metadata["run"]["status"] = "completed"
    metadata["run"]["completed_at"] = utc_now()
    _replace_metadata(records, metadata)
    metrics = aggregate_trials(records)
    derived = build_aggregate_result(metadata, metrics)
    findings = validate_result_pair(records, derived)
    if findings:
        first = findings[0]
        raise ValueError(f"canonical result failed validation with {first.code} {first.message}")
    write_jsonl(paths.raw, records)
    write_json(paths.derived, derived)
    return {"raw": records, "derived": derived, "paths": paths}


def run_single_experiment_cli(experiment_id: str) -> int:
    parser = argparse.ArgumentParser(description=f"Run the {experiment_id} experiment")
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_id = args.run_id or new_run_id()
    interface = FixtureModel() if args.fixture else None
    result = run_experiment(
        args.model, experiment_id, run_id=run_id, interface=interface,
        resume=args.resume, verbose=args.verbose,
        runner=f"src/tasks/{experiment_id}.py",
    )
    print(result["paths"].derived)
    return 0


def run_batch(
    model_id: str,
    *,
    run_id: str,
    experiment_ids: list[str] | None = None,
    fixture: bool = False,
    resume: bool = False,
    release_root: str | Path | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> list[dict[str, Any]]:
    selected = experiment_ids or [item["id"] for item in active_experiments()]
    results = []
    for experiment_id in selected:
        interface = FixtureModel() if fixture else None
        results.append(run_experiment(
            model_id, experiment_id, run_id=run_id, interface=interface,
            resume=resume, release_root=release_root, sleeper=sleeper,
        ))
    return results
