#!/usr/bin/env python3
"""Generate dashboard projections from canonical release results."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.results.aggregation import aggregate_trials
from src.results.dashboard import generate_dashboard_file
from src.results.io import read_json, read_jsonl


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
    parser.add_argument("--experiments", nargs="*")
    args = parser.parse_args()

    generated = []
    for derived_path in sorted((args.release_root / "derived").glob("*/*.json")):
        derived = read_json(derived_path)
        metadata = derived["metadata"]
        model_id = metadata["model"]["id"]
        experiment_id = metadata["experiment"]["id"]
        if args.models and model_id not in args.models:
            continue
        if args.experiments and experiment_id not in args.experiments:
            continue
        run_id = metadata["run"]["id"]
        model_key = derived_path.parent.name
        raw_path = (
            args.release_root
            / "raw"
            / model_key
            / experiment_id
            / f"{run_id}.jsonl"
        )
        if not raw_path.is_file():
            print(f"Missing raw input {raw_path}", file=sys.stderr)
            return 1
        raw_records = read_jsonl(raw_path)
        reproduced = aggregate_trials(raw_records)
        if reproduced != derived["aggregate_metrics"]:
            print(f"Aggregate mismatch {derived_path}", file=sys.stderr)
            return 1
        generated.append(generate_dashboard_file(raw_path, derived_path, args.output_dir))

    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
