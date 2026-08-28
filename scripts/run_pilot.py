#!/usr/bin/env python3
"""Preflight and launch the authorized one-model EconBench pilot."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from scripts.run_benchmark import main as run_benchmark_main
from scripts.smoke_models import main as smoke_models_main
from src.results.provenance import require_clean_repository
from src.tasks.config import active_experiments, model_config, validate_run_id
from src.tasks.engine import new_run_id


DEFAULT_PILOT_MODEL = "gemini-2.5-flash-lite"
PROVIDER_CREDENTIALS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight and run the complete one-model EconBench pilot through "
            "the canonical benchmark runner"
        )
    )
    parser.add_argument("--model", default=DEFAULT_PILOT_MODEL)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the one-call live endpoint check before collection",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate local prerequisites without making provider requests",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Exercise the full pilot workflow with the offline fixture provider",
    )
    return parser


def preflight(
    model_id: str,
    *,
    run_id: str,
    env_file: Path,
    fixture: bool,
) -> dict:
    """Validate all prerequisites that can be checked without provider access."""
    if sys.version_info < (3, 12):
        raise RuntimeError(
            "the pilot requires Python 3.12 or newer; "
            f"current interpreter is {sys.version.split()[0]}"
        )

    validate_run_id(run_id)
    model = model_config(model_id)
    if model is None or model["status"] != "active":
        raise RuntimeError(f"pilot model must be active and registered: {model_id!r}")

    if fixture:
        return model

    require_clean_repository(PROJECT_ROOT)
    if env_file.is_file():
        load_dotenv(env_file)

    credential = PROVIDER_CREDENTIALS.get(model["provider"])
    if credential is None:
        raise RuntimeError(
            f"no pilot credential mapping exists for provider {model['provider']!r}"
        )
    if not os.getenv(credential):
        raise RuntimeError(
            f"{credential} is not configured; set it in the environment or {env_file}"
        )
    return model


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or new_run_id().replace("run-", "pilot-", 1)

    try:
        model = preflight(
            args.model,
            run_id=run_id,
            env_file=args.env_file,
            fixture=args.fixture,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"PREFLIGHT FAIL: {exc}", file=sys.stderr)
        return 2

    experiment_count = len(active_experiments())
    mode = "offline fixture" if args.fixture else "live provider"
    print(
        f"PREFLIGHT PASS: model={args.model} endpoint={model['api_model_id']} "
        f"experiments={experiment_count} run_id={run_id} mode={mode}"
    )
    if args.preflight_only:
        return 0

    if not args.fixture and not args.skip_smoke:
        smoke_status = smoke_models_main([
            "--model",
            model["api_model_id"],
            "--max-new-tokens",
            "8",
            "--env-file",
            str(args.env_file),
        ])
        if smoke_status:
            print("Pilot aborted because the live endpoint smoke check failed.")
            return smoke_status

    benchmark_args = ["--model", args.model, "--run-id", run_id]
    if args.resume:
        benchmark_args.append("--resume")
    if args.fixture:
        benchmark_args.append("--fixture")
    return run_benchmark_main(benchmark_args)


if __name__ == "__main__":
    raise SystemExit(main())
