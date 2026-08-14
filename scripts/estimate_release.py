#!/usr/bin/env python3
"""Estimate release calls, tokens, elapsed time, and list price."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "econbench-matplotlib")
)

from src.tasks.config import active_experiments, experiment_config
from src.tasks.specs import bisection_conditions, bisection_plan, fixed_trial_plans


BISECTION_EXPERIMENTS = {"independence", "time"}
CHARACTERS_PER_TOKEN = 4
LOW_OUTPUT_TOKENS_PER_CALL = 8
HIGH_OUTPUT_TOKENS_PER_CALL = 128
STANDARD_LATENCY_SECONDS = (2, 10)
PRO_LATENCY_SECONDS = (60, 300)


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _bisection_prompts(config: dict[str, Any]) -> list[str]:
    prompts = []
    for base in bisection_conditions(config):
        trials = []
        while (plan := bisection_plan(config, base, trials)) is not None:
            prompts.append(plan.prompt)
            semantic_choice = (
                "sooner" if config["id"] == "time" else "reference_lottery"
            )
            trials.append({
                "condition": plan.condition,
                "trial_metrics": {"semantic_choice": semantic_choice},
                "validity": {"status": "valid"},
            })
    return prompts


def experiment_estimate(config: dict[str, Any]) -> dict[str, Any]:
    if config["id"] not in BISECTION_EXPERIMENTS:
        prompts = [plan.prompt for plan in fixed_trial_plans(config)]
        return {
            "id": config["id"],
            "primary_calls": len(prompts),
            "validation_calls": 0,
            "diagnostic_calls": 0,
            "total_calls": len(prompts),
            "estimated_prompt_characters": sum(map(len, prompts)),
        }

    settings = config["settings"]
    prompts = _bisection_prompts(config)
    conditions = len(bisection_conditions(config))
    iterations = settings["bisection_iterations"]
    repetitions = settings["responses_per_bisection_step"]
    validation_sequences = max(
        1, math.floor(conditions * settings["validation_fraction"])
    )
    validation_calls = validation_sequences * iterations * repetitions
    diagnostic_calls = settings["diagnostic_monotonicity_checks"]
    diagnostic_calls += settings.get("diagnostic_transitivity_checks", 0)
    diagnostic_calls += (
        settings["diagnostic_bidirectional_sequences"] * iterations * repetitions
    )
    total_calls = len(prompts) + validation_calls + diagnostic_calls
    mean_prompt_characters = sum(map(len, prompts)) / len(prompts)
    return {
        "id": config["id"],
        "primary_calls": len(prompts),
        "validation_calls": validation_calls,
        "diagnostic_calls": diagnostic_calls,
        "total_calls": total_calls,
        "estimated_prompt_characters": round(mean_prompt_characters * total_calls),
    }


def build_estimate() -> dict[str, Any]:
    models = _load(PROJECT_ROOT / "config" / "models.json")["models"]
    active_models = [model for model in models if model["status"] == "active"]
    pricing = _load(PROJECT_ROOT / "config" / "model_pricing.json")
    price_by_id = {record["id"]: record for record in pricing["models"]}
    experiments = [experiment_estimate(experiment_config(item["id"]))
                   for item in active_experiments()]
    calls_per_model = sum(item["total_calls"] for item in experiments)
    input_tokens_per_model = math.ceil(
        sum(item["estimated_prompt_characters"] for item in experiments)
        / CHARACTERS_PER_TOKEN
    )
    cost_rows = []
    for model in active_models:
        price = price_by_id[model["id"]]
        input_cost = input_tokens_per_model / 1_000_000 * price["input"]
        low_cost = input_cost + (
            calls_per_model * LOW_OUTPUT_TOKENS_PER_CALL / 1_000_000
            * price["output"]
        )
        high_cost = input_cost + (
            calls_per_model * HIGH_OUTPUT_TOKENS_PER_CALL / 1_000_000
            * price["output"]
        )
        cost_rows.append({
            "id": model["id"],
            "estimated_cost_low_usd": round(low_cost, 2),
            "estimated_cost_high_usd": round(high_cost, 2),
        })

    standard_models = [
        model for model in active_models if model["id"] != "gpt-5.2-pro"
    ]
    low_seconds = calls_per_model * (
        len(standard_models) * STANDARD_LATENCY_SECONDS[0]
        + PRO_LATENCY_SECONDS[0]
    )
    high_seconds = calls_per_model * (
        len(standard_models) * STANDARD_LATENCY_SECONDS[1]
        + PRO_LATENCY_SECONDS[1]
    )
    return {
        "benchmark_version": pricing["benchmark_version"],
        "reviewed_at": pricing["reviewed_at"],
        "active_models": len(active_models),
        "experiments": experiments,
        "calls_per_model": calls_per_model,
        "release_calls": calls_per_model * len(active_models),
        "estimated_input_tokens_per_model": input_tokens_per_model,
        "output_token_scenarios_per_call": {
            "low": LOW_OUTPUT_TOKENS_PER_CALL,
            "high": HIGH_OUTPUT_TOKENS_PER_CALL,
        },
        "estimated_cost_low_usd": round(sum(
            row["estimated_cost_low_usd"] for row in cost_rows
        ), 2),
        "estimated_cost_high_usd": round(sum(
            row["estimated_cost_high_usd"] for row in cost_rows
        ), 2),
        "estimated_serial_hours_low": round(low_seconds / 3600, 1),
        "estimated_serial_hours_high": round(high_seconds / 3600, 1),
        "model_costs": cost_rows,
        "assumptions": {
            "characters_per_input_token": CHARACTERS_PER_TOKEN,
            "standard_seconds_per_call": list(STANDARD_LATENCY_SECONDS),
            "gpt_5_2_pro_seconds_per_call": list(PRO_LATENCY_SECONDS),
            "retries_and_reruns_included": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    estimate = build_estimate()
    if args.as_json:
        print(json.dumps(estimate, indent=2))
    else:
        print(f"Active models {estimate['active_models']}")
        print(f"Calls per model {estimate['calls_per_model']}")
        print(f"Release calls {estimate['release_calls']}")
        print(
            "Estimated list price USD "
            f"{estimate['estimated_cost_low_usd']:.2f} to "
            f"{estimate['estimated_cost_high_usd']:.2f}"
        )
        print(
            "Estimated serial time hours "
            f"{estimate['estimated_serial_hours_low']:.1f} to "
            f"{estimate['estimated_serial_hours_high']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
