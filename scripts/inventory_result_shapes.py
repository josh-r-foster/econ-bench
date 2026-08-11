"""Inventory structural variants in EconBench JSON and JSONL result files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = (
    Path("data/results"),
    Path("web/data"),
    Path("web/public/data"),
)


def value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    raise TypeError(f"Unsupported scalar type {type(value).__name__}")


def shape_of(value: Any) -> Any:
    """Return a canonical nested description of JSON field names and value types."""
    if isinstance(value, dict):
        return {
            "object": {
                key: shape_of(item)
                for key, item in sorted(value.items())
            }
        }
    if isinstance(value, list):
        element_shapes = {
            json.dumps(shape_of(item), sort_keys=True, separators=(",", ":"))
            for item in value
        }
        return {"array": [json.loads(item) for item in sorted(element_shapes)]}
    return value_type(value)


def shape_id(shape: Any) -> str:
    encoded = json.dumps(shape, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def top_level_fields(values: Iterable[Any]) -> list[str]:
    fields: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            fields.update(value)
    return sorted(fields)


def result_family(path: Path) -> str:
    name = path.name
    parts = path.parts

    if name.endswith(".jsonl") and "runs" in parts:
        return "trace_log"
    if name == "models.json":
        return "model_registry"
    if name.startswith("independence_results_"):
        return "independence_dashboard"
    if name.startswith("time_experiment_"):
        return "time_dashboard"
    if name.endswith("_rationality.json"):
        return "rationality_dashboard"
    if name.startswith("social_experiment_"):
        return "social_legacy_dashboard"
    if name.endswith("_social_stats.json"):
        return "social_stats_dashboard"

    dashboard_prefixes = (
        "dictator",
        "ultimatum",
        "trust_game",
        "stag_hunt",
        "beauty_contest",
        "centipede_game",
        "public_goods",
        "travellers_dilemma",
        "matching_pennies",
    )
    for experiment_id in dashboard_prefixes:
        if name.startswith(f"{experiment_id}_experiment_"):
            return f"{experiment_id}_dashboard"

    if "data" in parts and "results" in parts:
        results_index = parts.index("results")
        if len(parts) > results_index + 1:
            experiment_id = parts[results_index + 1]
            return f"{experiment_id}_raw"

    return "unclassified"


def result_category(path: Path) -> str:
    parts = path.parts
    if "public" in parts:
        return "public_copy"
    if "web" in parts:
        return "dashboard"
    if path.suffix == ".jsonl":
        return "trace"
    return "raw"


def iter_result_files(paths: Sequence[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix in {".json", ".jsonl"}:
            files.add(path)
        elif path.is_dir():
            files.update(path.rglob("*.json"))
            files.update(path.rglob("*.jsonl"))
    return sorted(files)


def load_values(path: Path) -> list[Any]:
    if path.suffix == ".jsonl":
        values = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    values.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"line {line_number} has invalid JSON") from exc
        return values

    with path.open(encoding="utf-8") as handle:
        return [json.load(handle)]


def display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def inventory(paths: Sequence[Path], project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    files = iter_result_files(paths)
    file_records = []
    invalid_files = []
    jsonl_records = 0

    for path in files:
        shown_path = display_path(path, project_root)
        try:
            values = load_values(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            invalid_files.append({"path": shown_path, "error": str(exc)})
            continue

        if path.suffix == ".jsonl":
            jsonl_records += len(values)
            serialized_shapes = sorted({
                json.dumps(shape_of(value), sort_keys=True, separators=(",", ":"))
                for value in values
            })
            file_shape = {
                "jsonl_records": [json.loads(item) for item in serialized_shapes]
            }
        else:
            file_shape = shape_of(values[0])

        file_records.append(
            {
                "path": shown_path,
                "category": result_category(path),
                "family": result_family(path),
                "shape_id": shape_id(file_shape),
                "shape": file_shape,
                "top_level_type": (
                    "object"
                    if values and isinstance(values[0], dict)
                    else "array"
                    if values and isinstance(values[0], list)
                    else value_type(values[0])
                    if values
                    else "empty_jsonl"
                ),
                "top_level_fields": top_level_fields(values),
                "record_count": len(values) if path.suffix == ".jsonl" else 1,
            }
        )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in file_records:
        grouped[(record["category"], record["family"], record["shape_id"])].append(record)

    variants = []
    for (category, family, variant_id), records in sorted(grouped.items()):
        variants.append(
            {
                "category": category,
                "family": family,
                "shape_id": variant_id,
                "file_count": len(records),
                "top_level_types": sorted({record["top_level_type"] for record in records}),
                "top_level_fields": sorted({
                    field
                    for record in records
                    for field in record["top_level_fields"]
                }),
                "shape": records[0]["shape"],
                "paths": [record["path"] for record in records],
            }
        )

    return {
        "summary": {
            "files_discovered": len(files),
            "files_parsed": len(file_records),
            "json_files": sum(path.suffix == ".json" for path in files),
            "jsonl_files": sum(path.suffix == ".jsonl" for path in files),
            "jsonl_records": jsonl_records,
            "shape_variants": len(variants),
            "invalid_files": len(invalid_files),
            "unclassified_files": sum(
                record["family"] == "unclassified" for record in file_records
            ),
        },
        "variants": variants,
        "files": file_records,
        "errors": invalid_files,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=list(DEFAULT_PATHS),
        help="Files or directories to scan",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print only the summary object",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = [path if path.is_absolute() else PROJECT_ROOT / path for path in args.paths]
    report = inventory(paths)
    output = report["summary"] if args.summary else report
    print(json.dumps(output, indent=2, sort_keys=True))
    return 1 if report["errors"] or report["summary"]["unclassified_files"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
