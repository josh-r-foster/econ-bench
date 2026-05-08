"""
JSONL trace logger for EconBench model calls.

Appends one JSON line per generate_response call to:
  data/results/runs/{YYYYMMDD_HHMMSS}.jsonl   (default)

Override with ECONBENCH_LOG_DIR env var. Set to "none" to disable.
Tag the active experiment with ECONBENCH_EXPERIMENT env var.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_log_path: Optional[Path] = None
_initialized = False


def _resolve_log_path() -> Optional[Path]:
    global _log_path, _initialized
    if _initialized:
        return _log_path

    _initialized = True
    log_dir_env = os.environ.get("ECONBENCH_LOG_DIR", "data/results/runs")
    if log_dir_env.lower() == "none":
        _log_path = None
        return None

    log_dir = Path(log_dir_env)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _log_path = log_dir / f"session_{ts}.jsonl"
    except Exception:
        _log_path = None

    return _log_path


def log_event(event: Dict[str, Any]) -> None:
    """Append a JSON event to the current session log file. Never raises."""
    path = _resolve_log_path()
    if path is None:
        return
    event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    event.setdefault("experiment", os.environ.get("ECONBENCH_EXPERIMENT", "unknown"))
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass


def log_model_call(
    *,
    model: str,
    prompt_chars: int,
    response: str,
    latency_ms: float,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    valid: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Convenience wrapper for a single model API call event."""
    event: Dict[str, Any] = {
        "event": "model_call",
        "model": model,
        "prompt_chars": prompt_chars,
        "response_chars": len(response),
        "latency_ms": round(latency_ms, 1),
        "valid": valid,
    }
    if prompt_tokens is not None:
        event["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        event["completion_tokens"] = completion_tokens
    if extra:
        event.update(extra)
    log_event(event)
