#!/usr/bin/env python3
"""Generate rationality dashboard data from canonical release results."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.results.aggregation import aggregate_trials
from src.results.io import read_json, read_jsonl, write_json
from src.results.model_ids import model_id_from_path_component
from src.results.rationality import build_rationality_projection
from src.results.validation import validate_result_pair


def _load_pair(release_root: Path, model_key: str, experiment_id: str):
    derived_path = release_root / "derived" / model_key / f"{experiment_id}.json"
    if not derived_path.is_file():
        return None
    derived = read_json(derived_path)
    run_id = derived["metadata"]["run"]["id"]
    raw_path = release_root / "raw" / model_key / experiment_id / f"{run_id}.jsonl"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    raw = read_jsonl(raw_path)
    findings = validate_result_pair(raw, derived)
    if findings:
        first = findings[0]
        raise ValueError(f"{first.code} {first.location} {first.message}")
    if aggregate_trials(raw) != derived["aggregate_metrics"]:
        raise ValueError(f"aggregate mismatch for {model_key} {experiment_id}")
    return derived


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "releases" / "1.0.0",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "web" / "data"
    )
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args()

    derived_root = args.release_root / "derived"
    model_keys = sorted(path.name for path in derived_root.glob("*") if path.is_dir())
    generated = 0
    for model_key in model_keys:
        model_id = model_id_from_path_component(model_key)
        if args.models and model_id not in args.models:
            continue
        independence = _load_pair(args.release_root, model_key, "independence")
        time = _load_pair(args.release_root, model_key, "time")
        if independence is None or time is None:
            continue
        projection = build_rationality_projection(independence, time)
        output = args.output_dir / f"{model_key}_rationality.json"
        write_json(output, projection)
        print(output)
        generated += 1
    print(f"Generated {generated} rationality files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
