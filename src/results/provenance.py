"""Canonical timestamp, digest, and code revision helpers."""

from __future__ import annotations

import hashlib
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


_CODE_REVISION = re.compile(r"^[0-9a-f]{40}$")


def format_utc_timestamp(value: datetime) -> str:
    """Format an aware datetime as a UTC timestamp ending in Z."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    utc_value = value.astimezone(timezone.utc)
    return utc_value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def utc_now() -> str:
    """Return the current time in the canonical UTC representation."""
    return format_utc_timestamp(datetime.now(timezone.utc))


def normalize_timestamp(value: str, source_timezone: str | None = None) -> str:
    """Normalize an ISO timestamp and require a zone for naive legacy values."""
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp must be a nonempty string")

    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"invalid ISO timestamp {value!r}") from error

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        if source_timezone is None:
            raise ValueError("naive legacy timestamp requires source_timezone")
        try:
            parsed = parsed.replace(tzinfo=ZoneInfo(source_timezone))
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"unknown source timezone {source_timezone!r}") from error

    return format_utc_timestamp(parsed)


def normalize_code_revision(value: str) -> str:
    """Validate and return one lowercase forty character Git revision."""
    if not isinstance(value, str) or not _CODE_REVISION.fullmatch(value):
        raise ValueError("code revision must contain forty lowercase hexadecimal characters")
    return value


def text_sha256(value: str) -> str:
    """Return the lowercase SHA256 digest of UTF-8 text."""
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_provenance(repository: str | Path) -> tuple[str, bool]:
    """Return the current Git revision and dirty state for a repository."""
    root = Path(repository)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return normalize_code_revision(revision), bool(status.strip())


def require_clean_repository(repository: str | Path) -> str:
    """Return the revision of a clean repository or reject data collection."""
    revision, dirty = git_provenance(repository)
    if dirty:
        raise RuntimeError(
            "native benchmark collection requires a clean Git working tree"
        )
    return revision
