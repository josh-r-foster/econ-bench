#!/usr/bin/env python3
"""Regenerate dashboard and rationality projections from canonical results."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-root", type=Path,
        default=PROJECT_ROOT / "data" / "releases" / "1.0.0",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "web" / "data"
    )
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args(argv)

    common = [
        "--release-root", str(args.release_root),
        "--output-dir", str(args.output_dir),
    ]
    if args.models:
        common.extend(["--models", *args.models])
    commands = [
        [sys.executable, str(PROJECT_ROOT / "scripts" / "generate_dashboard_data.py"), *common],
        [sys.executable, str(PROJECT_ROOT / "src" / "tools" / "calculate_rationality_stats.py"), *common],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
