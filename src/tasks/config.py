"""Manifest backed task configuration and canonical output paths."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.results.model_ids import model_id_to_path_component


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_MANIFEST_PATH = PROJECT_ROOT / "config" / "experiments.json"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "config" / "models.json"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class RunPaths:
    raw: Path
    derived: Path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def experiment_manifest() -> dict[str, Any]:
    return _read_json(EXPERIMENT_MANIFEST_PATH)


@lru_cache(maxsize=1)
def model_manifest() -> dict[str, Any]:
    return _read_json(MODEL_MANIFEST_PATH)


def active_experiments() -> list[dict[str, Any]]:
    return [
        copy.deepcopy(item)
        for item in experiment_manifest()["experiments"]
        if item["status"] == "active"
    ]


def experiment_config(experiment_id: str) -> dict[str, Any]:
    for experiment in experiment_manifest()["experiments"]:
        if experiment["id"] == experiment_id and experiment["status"] == "active":
            return copy.deepcopy(experiment)
    raise ValueError(f"unknown active experiment {experiment_id!r}")


def model_config(model_id: str) -> dict[str, Any] | None:
    for model in model_manifest()["models"]:
        if model["id"] == model_id:
            return copy.deepcopy(model)
    return None


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run identifier must be a portable component with at most 128 characters"
        )
    return run_id


def canonical_run_paths(
    model_id: str,
    experiment_id: str,
    run_id: str,
    *,
    project_root: str | Path = PROJECT_ROOT,
    release_root: str | Path | None = None,
) -> RunPaths:
    """Resolve the canonical raw and derived paths for one run."""
    validate_run_id(run_id)
    model_key = model_id_to_path_component(model_id)
    if release_root is not None:
        root = Path(release_root)
        return RunPaths(
            raw=root / "raw" / model_key / experiment_id / f"{run_id}.jsonl",
            derived=root / "derived" / model_key / f"{experiment_id}.json",
        )

    manifest = experiment_manifest()
    locations = manifest["shared_settings"]["canonical_locations"]
    values = {
        "model_key": model_key,
        "experiment_id": experiment_id,
        "run_id": run_id,
    }
    root = Path(project_root)
    return RunPaths(
        raw=root / locations["raw"].format(**values),
        derived=root / locations["derived"].format(**values),
    )
