"""Shared model loading and response capture for experiment tasks."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from src.models.logger import log_event
from src.models.registry import get_model_interface
from src.results.provenance import text_sha256, utc_now


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_MANIFEST_PATH = PROJECT_ROOT / "config" / "models.json"


@dataclass(frozen=True)
class ModelBinding:
    """The benchmark and provider identities used for one model interface."""

    model_id: str
    api_model_id: str
    provider: str | None
    status: str | None
    registered: bool


@dataclass(frozen=True)
class CompletionResult:
    """One received completion or one exhausted provider failure."""

    status: str
    response: str | None
    logprobs: Any
    attempts: int
    started_at: str
    completed_at: str
    latency_ms: float
    error: dict[str, Any] | None


_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def is_retryable_transport_error(exception: Exception) -> bool:
    """Return whether a provider failure is safe to retry."""
    if isinstance(exception, (TimeoutError, ConnectionError)):
        return True
    name = type(exception).__name__.lower()
    if name.endswith(("timeouterror", "connectionerror")):
        return True
    status_code = getattr(exception, "status_code", None)
    if status_code is None:
        response = getattr(exception, "response", None)
        status_code = getattr(response, "status_code", None)
    return status_code in _RETRYABLE_STATUS_CODES


def resolve_model_binding(
    model_id: str, manifest_path: str | Path = MODEL_MANIFEST_PATH
) -> ModelBinding:
    """Resolve a benchmark model identifier to its provider endpoint."""
    with Path(manifest_path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    for model in manifest["models"]:
        if model["id"] == model_id:
            return ModelBinding(
                model_id=model_id,
                api_model_id=model["api_model_id"],
                provider=model["provider"],
                status=model["status"],
                registered=True,
            )

    return ModelBinding(
        model_id=model_id,
        api_model_id=model_id,
        provider=None,
        status=None,
        registered=False,
    )


def load_model_interface(
    model_id: str,
    *,
    loader: Callable[[str], Any] = get_model_interface,
    manifest_path: str | Path = MODEL_MANIFEST_PATH,
) -> Any:
    """Load one provider interface and retain both model identities on it."""
    load_dotenv()
    binding = resolve_model_binding(model_id, manifest_path)
    interface = loader(binding.api_model_id)
    interface.econbench_model_binding = binding
    return interface


def request_model_response(
    interface: Any,
    *,
    experiment_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    verbose: bool,
    return_logprobs: bool = False,
) -> str:
    """Request a response through the shared retry path."""
    completion = request_model_completion(
        interface,
        experiment_id=experiment_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        verbose=verbose,
        return_logprobs=return_logprobs,
    )
    return completion.response or ""


def request_model_completion(
    interface: Any,
    *,
    experiment_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    verbose: bool,
    return_logprobs: bool = False,
    maximum_retries: int = 2,
    backoff_seconds: tuple[float, ...] = (2, 4),
    sleeper: Callable[[float], None] = time.sleep,
) -> CompletionResult:
    """Request one completion with bounded transport retries and full capture."""
    if maximum_retries < 0:
        raise ValueError("maximum_retries must not be negative")
    if len(backoff_seconds) < maximum_retries:
        raise ValueError("retry backoff does not cover every retry")
    binding = getattr(
        interface,
        "econbench_model_binding",
        ModelBinding(
            model_id=getattr(interface, "model_id", "unknown"),
            api_model_id=getattr(interface, "model_id", "unknown"),
            provider=None,
            status=None,
            registered=False,
        ),
    )
    started_at = utc_now()
    started_clock = time.perf_counter()
    response: str | None = None
    logprobs = None
    error: dict[str, Any] | None = None
    attempts = 0

    for attempt in range(maximum_retries + 1):
        attempts = attempt + 1
        try:
            result = interface.generate_response(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                return_logprobs=return_logprobs,
                verbose=verbose,
            )
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError(
                    "model interface must return a response and log probabilities"
                )
            response, logprobs = result
            if not isinstance(response, str):
                response = None
                raise TypeError("model interface response must be a string")
            error = None
            break
        except Exception as exception:
            retryable = is_retryable_transport_error(exception)
            error = {
                "type": type(exception).__name__,
                "message": str(exception),
                "retryable": retryable,
            }
            if not retryable:
                break
            if attempt < maximum_retries:
                sleeper(backoff_seconds[attempt])

    completed_at = utc_now()
    latency_ms = round((time.perf_counter() - started_clock) * 1000, 3)
    event = {
        "event": "task_model_response",
        "experiment": experiment_id,
        "model": binding.model_id,
        "api_model_id": binding.api_model_id,
        "provider": binding.provider,
        "started_at": started_at,
        "completed_at": completed_at,
        "latency_ms": latency_ms,
        "attempts": attempts,
        "request": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_logprobs": return_logprobs,
        },
        "prompt": prompt,
        "prompt_sha256": text_sha256(prompt),
        "response": response,
        "response_sha256": text_sha256(response) if response is not None else None,
        "response_valid": bool(response),
        "logprobs": logprobs,
        "error": error,
    }
    log_event(event)
    return CompletionResult(
        status="received" if error is None else "provider_error",
        response=response,
        logprobs=logprobs,
        attempts=attempts,
        started_at=started_at,
        completed_at=completed_at,
        latency_ms=latency_ms,
        error=error,
    )


def model_binding_dict(interface: Any) -> dict[str, Any]:
    """Return a serializable copy of the binding retained on an interface."""
    binding = getattr(interface, "econbench_model_binding", None)
    if binding is None:
        raise ValueError("model interface has no EconBench model binding")
    return asdict(binding)
