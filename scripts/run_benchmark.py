#!/usr/bin/env python3
"""Run one or more canonical EconBench experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks.config import active_experiments
from src.tasks.engine import new_run_id, run_batch


def main(argv: list[str] | None = None) -> int:
    active = [item["id"] for item in active_experiments()]
    parser = argparse.ArgumentParser(description="Run canonical EconBench experiments")
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--experiments", nargs="+", choices=active)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--fixture", action="store_true",
        help="Use the deterministic offline provider",
    )
    parser.add_argument("--release-root", type=Path)
    args = parser.parse_args(argv)

    run_id = args.run_id or new_run_id()
    results = run_batch(
        args.model, run_id=run_id, experiment_ids=args.experiments,
        fixture=args.fixture, resume=args.resume, release_root=args.release_root,
    )
    for result in results:
        print(result["paths"].derived)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
