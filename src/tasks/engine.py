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
from src.results.provenance import (
    git_provenance,
    require_clean_repository,
    utc_now,
)
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
from src.models.inference_controls import recorded_inference_controls
from src.tasks.specs import (
    BISECTION_EXPERIMENTS,
    TrialPlan,
    fixed_trial_plans,
    next_trial_plan,
)


class FixtureModel:
    """Deterministic offline provider used by the complete batch test."""

    model_id = "offline-fixture"

    def generate_response(self, *, prompt: str, **_kwargs):
        upper = prompt.upper()
        option_lines = {
            label: match.group(1)
            for label in ("A", "B")
            if (
                match := re.search(
                    rf"^OPTION {label}: (.+)$",
                    prompt,
                    flags=re.MULTILINE | re.IGNORECASE,
                )
            )
        }
        if len(option_lines) == 2 and "% chance of $" in prompt:
            def expected_value(description: str) -> float:
                return sum(
                    float(probability) / 100 * float(outcome)
                    for probability, outcome in re.findall(
                        r"([0-9.]+)% chance of \$([0-9.]+)", description
                    )
                )

            values = {
                label: expected_value(description)
                for label, description in option_lines.items()
            }
            return f"CHOICE={'A' if values['A'] >= values['B'] else 'B'}", None
        if len(option_lines) == 2 and "payment options" in prompt:
            def payment(description: str) -> tuple[float, int]:
                match = re.fullmatch(
                    r"\$([0-9.]+) after ([0-9]+) days", description
                )
                return float(match.group(1)), int(match.group(2))

            payments = {
                label: payment(description)
                for label, description in option_lines.items()
            }
            choice = max(
                payments,
                key=lambda label: (payments[label][0], -payments[label][1]),
            )
            return f"CHOICE={choice}", None
        if "DECISION=ACCEPT OR DECISION=REJECT" in upper:
            return "DECISION=ACCEPT", None
        if "CHOICE=A OR CHOICE=B" in upper:
            return "CHOICE=A", None
        if "CHOICE=HEADS OR CHOICE=TAILS" in upper:
            return "CHOICE=HEADS", None
        if "ACTION=PASS OR ACTION=TAKE" in upper:
            return "ACTION=TAKE", None
        if "CHOICE=<WHOLE NUMBER>" in upper:
            return "CHOICE=0", None
        if match := re.search(
            r"DOLLAR CLAIM FROM \$([0-9.]+) TO \$([0-9.]+)", upper
        ):
            return f"CLAIM={match.group(1)}", None
        if "TRANSFER=<AMOUNT>" in upper:
            return "TRANSFER=0", None
        if "OFFER=<AMOUNT>" in upper:
            return "OFFER=0", None
        if "SEND=<AMOUNT>" in upper:
            return "SEND=0", None
        if "RETURN=<AMOUNT>" in upper:
            return "RETURN=0", None
        if "CONTRIBUTION=<AMOUNT>" in upper:
            return "CONTRIBUTION=0", None
        return "INVALID", None


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
    capture_method: str = "native",
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
    if (
        model["provider"] == "anthropic"
        and model["api_model_id"] == "claude-opus-4-7"
    ):
        effective_temperature = None
    controls = recorded_inference_controls(
        model["provider"], model["api_model_id"]
    )
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
                "requested_reasoning_mode": controls["requested_reasoning_mode"],
                "effective_reasoning_mode": controls["effective_reasoning_mode"],
                "seed": shared["local_random_seed"],
                "system_prompt": shared["system_prompt"],
                "tools_enabled": shared["tools_enabled"],
                "provider_options": controls["provider_options"],
            },
        },
        "protocol": {
            "condition_order": shared["condition_order"],
            "local_random_seed": shared["local_random_seed"],
            "order_seed": (
                f"{shared['local_random_seed']}:{model_id}:{config['id']}"
            ),
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
            "capture_method": capture_method,
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


def _trial_from_completion(plan: TrialPlan, sequence_index: int, completion) -> dict[str, Any]:
    if completion.status == "provider_error":
        retryable = bool(completion.error and completion.error.get("retryable"))
        error = {
            "category": "provider",
            "code": completion.error["type"] if completion.error else None,
            "message": completion.error["message"] if completion.error else "Provider failure",
            "retryable": retryable,
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
            validity_reason_code=(
                "provider_attempts_exhausted"
                if retryable
                else "provider_nonretryable_error"
            ),
            validity_reason=(
                "Provider call failed after bounded transport retries"
                if retryable
                else "Provider call failed with a nonretryable error"
            ),
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
    capture_method = "fixture" if interface is not None else "native"
    if capture_method == "native":
        require_clean_repository(root)
    if paths.raw.exists() and not resume:
        raise FileExistsError(f"raw run already exists at {paths.raw}")

    records = read_jsonl(paths.raw) if paths.raw.exists() else []
    if records:
        metadata = copy.deepcopy(records[0]["metadata"])
        expected_metadata = _metadata(
            model_id, config, run_id, metadata["run"]["started_at"],
            runner=runner, project_root=root, capture_method=capture_method,
        )
        for field in (
            "benchmark_version", "schema_version", "experiment", "model", "protocol"
        ):
            if metadata[field] != expected_metadata[field]:
                raise ValueError(f"resume metadata has stale {field}")
        for field in (
            "capture_method", "code_revision", "repository_dirty", "runner",
            "python_version", "platform",
        ):
            if metadata["provenance"][field] != expected_metadata["provenance"][field]:
                raise ValueError(f"resume metadata has stale provenance {field}")
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
            model_id, config, run_id, utc_now(), runner=runner, project_root=root,
            capture_method=capture_method,
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

    order_seed = metadata["protocol"]["order_seed"]
    fixed = (
        []
        if experiment_id in BISECTION_EXPERIMENTS
        else fixed_trial_plans(config, order_seed)
    )
    retry = metadata["protocol"]["transport_retry_policy"]
    sleeper_function = sleeper if sleeper is not None else __import__("time").sleep

    while (plan := next_trial_plan(config, records, fixed, order_seed)) is not None:
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
    if not args.fixture:
        require_clean_repository(PROJECT_ROOT)
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
    if not fixture:
        require_clean_repository(PROJECT_ROOT)
    selected = experiment_ids or [item["id"] for item in active_experiments()]
    results = []
    for experiment_id in selected:
        interface = FixtureModel() if fixture else None
        results.append(run_experiment(
            model_id, experiment_id, run_id=run_id, interface=interface,
            resume=resume, release_root=release_root, sleeper=sleeper,
            runner="fixture" if fixture else "scripts/run_benchmark.py",
        ))
    return results
