"""
Four-layer quality check for a completed EconBench run.

Reads JSONL trace logs from data/results/runs/ and result JSON from web/data/
to produce a pass/fail report across four check layers:

  1. Outcome    — % valid (non-empty) model responses
  2. Process    — bisection convergence in independence results
  3. Style      — all expected result files present and non-empty
  4. Efficiency — token usage relative to cross-model median

Usage:
    python scripts/check_run.py --model gpt-4o
    python scripts/check_run.py --model gpt-4o --log-dir data/results/runs
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Resolve project root so this script works from any cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict]:
    events = []
    if not path.exists():
        return events
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def _find_latest_log(model: str, log_dir: Path) -> Optional[Path]:
    """Return the most-recent session JSONL that contains calls for `model`."""
    if not log_dir.exists():
        return None
    candidates = sorted(log_dir.glob("session_*.jsonl"), reverse=True)
    for path in candidates:
        events = _load_jsonl(path)
        if any(e.get("model") == model for e in events):
            return path
    return None


def _fmt(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{100 * n / total:.1f}%"


# ---------------------------------------------------------------------------
# Check 1 — Outcome: valid response rate from trace log
# ---------------------------------------------------------------------------

def check_outcome(model: str, log_dir: Path) -> Tuple[bool, str]:
    log_file = _find_latest_log(model, log_dir)
    if log_file is None:
        return False, f"No trace log found for {model} in {log_dir}"

    events = [e for e in _load_jsonl(log_file) if e.get("event") == "model_call" and e.get("model") == model]
    if not events:
        return False, "No model_call events found in trace log"

    valid = sum(1 for e in events if e.get("valid", False))
    total = len(events)
    rate = valid / total if total else 0
    threshold = 0.95
    ok = rate >= threshold
    detail = f"valid responses: {valid}/{total} ({_pct(valid, total)}) — threshold ≥{int(threshold*100)}%"
    return ok, detail


# ---------------------------------------------------------------------------
# Check 2 — Process: bisection convergence in independence results
# ---------------------------------------------------------------------------

def check_process(model: str) -> Tuple[bool, str]:
    result_path = PROJECT_ROOT / "web" / "data" / f"independence_results_{model}.json"
    if not result_path.exists():
        return False, f"Missing {result_path.name}"

    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        return False, "No results array in independence JSON"

    target_iters = 10
    converged = [r for r in results if r.get("n_iterations", 0) >= target_iters]
    total = len(results)
    rate = len(converged) / total if total else 0
    threshold = 0.80
    ok = rate >= threshold
    detail = f"bisection converged ({target_iters} iters): {len(converged)}/{total} ({_pct(len(converged), total)}) — threshold ≥{int(threshold*100)}%"
    return ok, detail


# ---------------------------------------------------------------------------
# Check 3 — Style: expected result files present and non-empty
# ---------------------------------------------------------------------------

EXPECTED_FILES = [
    "web/data/independence_results_{model}.json",
    "web/data/{model}_rationality.json",
    "web/data/{model}_social_stats.json",
    "data/results/independence/{model}/mm_triangle_results.json",
    "data/results/time/{model}",
]


def check_style(model: str) -> Tuple[bool, str]:
    missing = []
    for template in EXPECTED_FILES:
        path = PROJECT_ROOT / template.format(model=model)
        if not path.exists():
            missing.append(path.name if path.suffix else path.parts[-1])
        elif path.is_file() and path.stat().st_size == 0:
            missing.append(f"{path.name} (empty)")

    ok = len(missing) == 0
    if ok:
        detail = f"all {len(EXPECTED_FILES)} expected paths present"
    else:
        detail = f"missing: {', '.join(missing)}"
    return ok, detail


# ---------------------------------------------------------------------------
# Check 4 — Efficiency: token usage vs. cross-model median
# ---------------------------------------------------------------------------

def check_efficiency(model: str, log_dir: Path) -> Tuple[bool, str]:
    if not log_dir.exists():
        return True, "no trace logs — skipping efficiency check"

    # Collect total tokens per model across all session logs
    model_totals: Dict[str, int] = {}
    for log_file in log_dir.glob("session_*.jsonl"):
        for event in _load_jsonl(log_file):
            if event.get("event") != "model_call":
                continue
            m = event.get("model", "")
            pt = event.get("prompt_tokens") or 0
            ct = event.get("completion_tokens") or 0
            model_totals[m] = model_totals.get(m, 0) + pt + ct

    if model in model_totals and len(model_totals) >= 2:
        others = [v for k, v in model_totals.items() if k != model and v > 0]
        if others:
            median = sorted(others)[len(others) // 2]
            this = model_totals[model]
            ratio = this / median if median else 1.0
            ok = ratio <= 2.0
            detail = f"total tokens: {this:,} vs. peer median {median:,} (ratio {ratio:.2f}x) — threshold ≤2.0x"
            return ok, detail

    # Not enough data to compare
    total = model_totals.get(model, 0)
    if total > 0:
        return True, f"total tokens logged: {total:,} (no cross-model comparison yet)"
    return True, "no token data in trace logs — skipping efficiency check"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Four-layer quality check for an EconBench run")
    parser.add_argument("--model", required=True, help="Model ID (e.g. gpt-4o)")
    parser.add_argument(
        "--log-dir",
        default=str(PROJECT_ROOT / "data" / "results" / "runs"),
        help="Directory containing session JSONL trace logs",
    )
    args = parser.parse_args()
    log_dir = Path(args.log_dir)

    checks = [
        ("Outcome  ", check_outcome(args.model, log_dir)),
        ("Process  ", check_process(args.model)),
        ("Style    ", check_style(args.model)),
        ("Efficiency", check_efficiency(args.model, log_dir)),
    ]

    print(f"\nEconBench run quality report — model: {args.model}")
    print("=" * 60)
    all_pass = True
    for label, (ok, detail) in checks:
        status = _fmt(ok)
        print(f"  [{status}] {label}: {detail}")
        if not ok:
            all_pass = False

    print("=" * 60)
    overall = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"
    print(f"  {overall}\n")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
