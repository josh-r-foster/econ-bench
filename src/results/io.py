"""Read and write canonical result records."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def read_json(path: str | Path) -> Any:
    """Read one UTF-8 JSON document."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read nonempty JSONL lines as objects."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(value)
    return records


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_json(path: str | Path, value: Any) -> None:
    """Write one JSON document atomically with stable formatting."""
    text = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    _atomic_text_write(Path(path), text)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Write JSON objects as atomic stable JSONL."""
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    text = "\n".join(lines)
    if lines:
        text += "\n"
    _atomic_text_write(Path(path), text)
