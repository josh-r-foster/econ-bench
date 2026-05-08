"""
Continuous GC script: validate results completeness and schema for all registered models.

Reads web/data/models.json for the registered model list, then checks that each model
has the required result files with valid structure. Prints a table and exits non-zero
if any model has missing or invalid data.

Usage:
    python scripts/validate_results.py
    python scripts/validate_results.py --models gpt-4o claude-sonnet-4-5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DATA = PROJECT_ROOT / "web" / "data"


# ---------------------------------------------------------------------------
# Schema checks per file type
# ---------------------------------------------------------------------------

def _check_independence(model: str) -> Tuple[str, str]:
    path = WEB_DATA / f"independence_results_{model}.json"
    if not path.exists():
        return "MISSING", str(path.name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return "INVALID", f"JSON parse error: {e}"
    results = data.get("results")
    if not isinstance(results, list) or len(results) == 0:
        return "INVALID", "missing or empty 'results' array"
    # Check for null indifference values
    nulls = sum(1 for r in results if r.get("indifference_value") is None)
    if nulls > len(results) * 0.5:
        return "WARN", f"{nulls}/{len(results)} null indifference values"
    return "PASS", f"{len(results)} results"


def _check_rationality(model: str) -> Tuple[str, str]:
    path = WEB_DATA / f"{model}_rationality.json"
    if not path.exists():
        return "MISSING", str(path.name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return "INVALID", f"JSON parse error: {e}"
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        return "INVALID", "missing 'metrics' object"
    required_keys = {"patience", "risk"}
    missing = required_keys - set(metrics.keys())
    if missing:
        return "WARN", f"missing metric keys: {missing}"
    return "PASS", "ok"


def _check_social(model: str) -> Tuple[str, str]:
    path = WEB_DATA / f"{model}_social_stats.json"
    if not path.exists():
        return "MISSING", str(path.name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return "INVALID", f"JSON parse error: {e}"
    if not isinstance(data, dict) or len(data) == 0:
        return "INVALID", "empty or non-object JSON"
    return "PASS", "ok"


CHECKS = [
    ("independence", _check_independence),
    ("rationality", _check_rationality),
    ("social_stats", _check_social),
]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

STATUS_ICONS = {"PASS": "✓", "WARN": "~", "MISSING": "✗", "INVALID": "✗"}
COL_WIDTH = 14


def _cell(status: str) -> str:
    icon = STATUS_ICONS.get(status, "?")
    return f"{icon} {status}".ljust(COL_WIDTH)


def _load_registered_models() -> List[str]:
    models_path = WEB_DATA / "models.json"
    if not models_path.exists():
        print(f"Error: {models_path} not found. Cannot determine registered models.")
        sys.exit(1)
    return json.loads(models_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate EconBench results completeness")
    parser.add_argument("--models", nargs="*", help="Specific model IDs to check (default: all in models.json)")
    args = parser.parse_args()

    models = args.models if args.models else _load_registered_models()
    check_names = [name for name, _ in CHECKS]

    # Header
    model_col = 30
    print(f"\n{'Model'.ljust(model_col)}" + "".join(n.ljust(COL_WIDTH) for n in check_names))
    print("-" * (model_col + COL_WIDTH * len(CHECKS)))

    overall_ok = True
    results_by_model: Dict[str, List[Tuple[str, str]]] = {}

    for model in models:
        row_results = []
        for _, check_fn in CHECKS:
            status, detail = check_fn(model)
            row_results.append((status, detail))
            if status in ("MISSING", "INVALID"):
                overall_ok = False

        results_by_model[model] = row_results
        display_name = model[:model_col - 1].ljust(model_col)
        print(display_name + "".join(_cell(s) for s, _ in row_results))

    print("-" * (model_col + COL_WIDTH * len(CHECKS)))

    # Detail section for failures
    failures = []
    for model, row_results in results_by_model.items():
        for (check_name, _), (status, detail) in zip(CHECKS, row_results):
            if status in ("MISSING", "INVALID", "WARN"):
                failures.append((model, check_name, status, detail))

    if failures:
        print("\nDetails:")
        for model, check, status, detail in failures:
            print(f"  [{status}] {model} / {check}: {detail}")

    verdict = "All models PASS" if overall_ok else "Some models have missing or invalid results"
    print(f"\n{verdict}\n")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
