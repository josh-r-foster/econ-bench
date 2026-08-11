"""Builders for canonical trial and aggregate result records."""

from __future__ import annotations

import copy
from typing import Any

from .provenance import normalize_timestamp, text_sha256


VALIDITY_STATES = {
    "valid",
    "invalid_response",
    "provider_error",
    "interrupted",
}


def build_trial(
    *,
    trial_id: str,
    sequence_index: int,
    condition_id: str,
    condition: dict[str, Any],
    repetition: int,
    role: str | None,
    started_at: str,
    completed_at: str,
    prompt_text: str,
    raw_response: str | None,
    parser_name: str,
    parser_status: str,
    parsed_value: Any,
    validity_status: str,
    trial_metrics: dict[str, Any],
    parser_error_code: str | None = None,
    parser_error_message: str | None = None,
    validity_reason_code: str | None = None,
    validity_reason: str | None = None,
    provider_request_id: str | None = None,
    finish_reason: str | None = None,
    attempts: int = 1,
    latency_ms: float | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one canonical trial while retaining complete interaction fields."""
    if validity_status not in VALIDITY_STATES:
        raise ValueError(f"unsupported validity status {validity_status!r}")
    if not prompt_text:
        raise ValueError("prompt_text must not be empty")
    if validity_status != "valid" and trial_metrics:
        raise ValueError("nonvalid trials cannot contain substantive metrics")

    prompt = {
        "text": prompt_text,
        "sha256": text_sha256(prompt_text),
    }
    response = {
        "raw_text": raw_response,
        "sha256": text_sha256(raw_response) if raw_response is not None else None,
        "provider_request_id": provider_request_id,
        "finish_reason": finish_reason,
    }

    return {
        "trial_id": trial_id,
        "sequence_index": sequence_index,
        "condition_id": condition_id,
        "condition": copy.deepcopy(condition),
        "repetition": repetition,
        "role": role,
        "started_at": normalize_timestamp(started_at),
        "completed_at": normalize_timestamp(completed_at),
        "prompt": prompt,
        "response": response,
        "parser": {
            "name": parser_name,
            "status": parser_status,
            "parsed_value": copy.deepcopy(parsed_value),
            "error_code": parser_error_code,
            "error_message": parser_error_message,
        },
        "validity": {
            "status": validity_status,
            "reason_code": validity_reason_code,
            "reason": validity_reason,
        },
        "transport": {
            "attempts": attempts,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "error": copy.deepcopy(error),
        "trial_metrics": copy.deepcopy(trial_metrics),
    }


def build_trial_result(metadata: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    """Wrap one trial with canonical metadata."""
    return {
        "record_type": "trial",
        "metadata": copy.deepcopy(metadata),
        "trial": copy.deepcopy(trial),
        "aggregate_metrics": None,
    }


def build_aggregate_result(
    metadata: dict[str, Any], aggregate_metrics: dict[str, Any]
) -> dict[str, Any]:
    """Wrap aggregate metrics with canonical metadata."""
    if not aggregate_metrics:
        raise ValueError("aggregate_metrics must not be empty")
    return {
        "record_type": "aggregate",
        "metadata": copy.deepcopy(metadata),
        "trial": None,
        "aggregate_metrics": copy.deepcopy(aggregate_metrics),
    }
