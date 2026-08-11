#!/usr/bin/env python3
"""Migrate one legacy combined social result into canonical split records."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.results.social_migration import migrate_legacy_social, write_social_migration
from src.results.validation import validate_result_pair


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "releases" / "1.0.0",
    )
    parser.add_argument("--source-timezone", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    migrations = migrate_legacy_social(
        source,
        source_path=str(args.source),
        source_timezone=args.source_timezone,
        project_root=PROJECT_ROOT,
    )
    findings = []
    for payload in migrations.values():
        findings.extend(validate_result_pair(payload["raw"], payload["derived"]))
    if findings:
        for finding in findings:
            print(f"{finding.code} {finding.location} {finding.message}")
        return 1

    for path in write_social_migration(
        migrations, args.output_root, overwrite=args.overwrite
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
