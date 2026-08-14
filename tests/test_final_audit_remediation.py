from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import httpx
import pytest
from google.genai import errors as google_errors

from scripts.validate_results import _cell_status
from src.results.aggregation import aggregate_trials
from src.results.io import write_json, write_jsonl
from src.results.validation import validate_result_pair
from src.tasks import engine
from src.tasks.config import experiment_config
from src.tasks.runtime import request_model_completion
from src.tasks.specs import bisection_conditions, bisection_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _reduced_time_config(experiment_id):
    config = experiment_config(experiment_id)
    settings = config["settings"]
    settings["amounts"] = settings["amounts"][:1]
    settings["delay_months"] = settings["delay_months"][:1]
    settings["front_end_delay_months"] = settings["front_end_delay_months"][:1]
    settings["bisection_iterations"] = 2
    settings["diagnostic_monotonicity_checks"] = 0
    settings["diagnostic_bidirectional_sequences"] = 0
    return config


def test_release_rejects_fixture_capture_and_truncated_complete_plan(tmp_path):
    result = engine.run_experiment(
        "gpt-5.2",
        "dictator",
        run_id="release-integrity",
        interface=engine.FixtureModel(),
        release_root=tmp_path,
        sleeper=lambda _seconds: None,
    )
    assert _cell_status(tmp_path, "gpt-5.2", "dictator")["status"] == "INVALID"

    noncanonical = copy.deepcopy(result)
    for record in [*noncanonical["raw"], noncanonical["derived"]]:
        provenance = record["metadata"]["provenance"]
        provenance["capture_method"] = "native"
        provenance["repository_dirty"] = False
        provenance["runner"] = "src/tasks/dictator.py"
    write_jsonl(noncanonical["paths"].raw, noncanonical["raw"])
    write_json(noncanonical["paths"].derived, noncanonical["derived"])
    status = _cell_status(tmp_path, "gpt-5.2", "dictator")
    assert status["status"] == "INVALID"
    assert "canonical runner" in status["detail"]

    truncated = copy.deepcopy(result["raw"][:-1])
    derived = copy.deepcopy(result["derived"])
    derived["aggregate_metrics"] = aggregate_trials(truncated)
    codes = {
        finding.code for finding in validate_result_pair(truncated, derived)
    }
    assert "incomplete_trial_plan" in codes


def test_canonical_validation_rejects_prompt_condition_transition_tampering(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(engine, "experiment_config", _reduced_time_config)
    result = engine.run_experiment(
        "gpt-5.2",
        "time",
        run_id="condition-binding",
        interface=engine.FixtureModel(),
        release_root=tmp_path,
        sleeper=lambda _seconds: None,
    )
    tampered = copy.deepcopy(result["raw"])
    target = next(
        record for record in tampered
        if record["trial"]["condition"].get("stage") == "bisection"
    )
    condition = target["trial"]["condition"]
    condition["midpoint"] = round(
        (condition["midpoint"] + condition["upper_bound_before"]) / 2, 2
    )
    derived = copy.deepcopy(result["derived"])
    derived["aggregate_metrics"] = aggregate_trials(tampered)
    codes = {finding.code for finding in validate_result_pair(tampered, derived)}
    assert "canonical_plan_mismatch" in codes


def test_time_records_censoring_instead_of_a_forced_finite_rate(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(engine, "experiment_config", _reduced_time_config)

    class AlwaysLater:
        def generate_response(self, *, prompt, **_kwargs):
            options = {
                label: int(days)
                for label, days in re.findall(
                    r"Option ([AB]): \$[0-9.]+ after ([0-9]+) days", prompt
                )
            }
            return f"CHOICE={max(options, key=options.get)}", None

    result = engine.run_experiment(
        "gpt-5.2",
        "time",
        run_id="right-censored",
        interface=AlwaysLater(),
        release_root=tmp_path,
        sleeper=lambda _seconds: None,
    )
    primary = result["derived"]["aggregate_metrics"]["discount_estimates"][0]
    assert primary["sequence_status"] == "right_censored"
    assert primary["indifference_lower_bound"] == primary["larger_amount"]
    assert primary["indifference_upper_bound"] is None
    assert primary["indifference_amount"] is None
    assert primary["discount_factor"] is None
    assert all(
        fit["status"] == "insufficient_data"
        for fit in result["derived"]["aggregate_metrics"]["model_fits"]
    )


def _lottery_utility(description):
    utility = {0.0: 0.0, 500.0: 0.9, 1000.0: 1.0}
    return sum(
        float(probability) / 100 * utility[float(outcome)]
        for probability, outcome in re.findall(
            r"([0-9.]+)% chance of \$([0-9.]+)", description
        )
    )


def test_independence_chooses_an_observed_bracket_for_concave_expected_utility():
    config = experiment_config("independence")
    config["settings"]["bisection_iterations"] = 2
    base = next(
        condition
        for condition in bisection_conditions(config, "concave-eu")
        if condition["reference_p_middle"] == 0
        and condition["reference_p_low"] == pytest.approx(5 / 12, abs=0.001)
    )
    assert "axis" not in base
    trials = []
    plans = []
    while plan := bisection_plan(config, base, trials):
        plans.append(plan)
        option_a, option_b = re.findall(r"Option [AB]: (.+)", plan.prompt)
        choice = "A" if _lottery_utility(option_a) >= _lottery_utility(option_b) else "B"
        parsed = plan.parser(f"CHOICE={choice}")
        trials.append({
            "condition": plan.condition,
            "trial_metrics": parsed.metrics,
            "validity": {"status": "valid"},
        })

    bisection = [plan for plan in plans if plan.condition["stage"] == "bisection"]
    assert bisection
    assert {plan.condition["axis"] for plan in bisection} == {"x"}
    assert all(plan.condition["reference_p_low"] > 0 for plan in plans)
    assert all(plan.condition["reference_p_high"] > 0 for plan in plans)


def test_displayed_elicitation_values_equal_recorded_treatments(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "experiment_config", _reduced_time_config)
    result = engine.run_experiment(
        "gpt-5.2",
        "time",
        run_id="display-integrity",
        interface=engine.FixtureModel(),
        release_root=tmp_path,
        sleeper=lambda _seconds: None,
    )
    for record in result["raw"]:
        trial = record["trial"]
        condition = trial["condition"]
        if "sooner_amount" not in condition:
            continue
        assert f"${condition['sooner_amount']:.2f}" in trial["prompt"]["text"]
        assert f"${condition['later_amount']:.2f}" in trial["prompt"]["text"]


@pytest.mark.parametrize(
    "exception",
    [
        httpx.ReadTimeout("read timeout"),
        httpx.ConnectTimeout("connect timeout"),
        httpx.ConnectError("connect error"),
        google_errors.ServerError(503, {"message": "unavailable"}, response=None),
    ],
)
def test_real_provider_transport_exceptions_receive_three_attempts(exception):
    class FailingInterface:
        calls = 0

        def generate_response(self, **_kwargs):
            self.calls += 1
            raise exception

    interface = FailingInterface()
    completion = request_model_completion(
        interface,
        experiment_id="dictator",
        prompt="prompt",
        max_new_tokens=8,
        temperature=0.5,
        verbose=False,
        maximum_retries=2,
        backoff_seconds=(0, 0),
        sleeper=lambda _seconds: None,
    )
    assert interface.calls == 3
    assert completion.attempts == 3
    assert completion.error["retryable"] is True


def test_openai_deprecation_ledger_controls_release_status():
    models = json.loads((PROJECT_ROOT / "config/models.json").read_text())
    matrix = json.loads((PROJECT_ROOT / "config/release_matrix.json").read_text())
    availability = json.loads(
        (PROJECT_ROOT / "config/model_availability.json").read_text()
    )
    by_id = {model["id"]: model for model in models["models"]}
    reviewed = {model["id"] for model in availability["models"]}
    assert by_id["o3"]["status"] == "retired"
    assert "deprecated" in by_id["o3"]["retired_reason"].lower()
    assert set(matrix["matrix"]["o3"].values()) == {"excluded"}
    assert "o3" not in reviewed

    assert by_id["gpt-4o"]["status"] == "active"
    assert set(matrix["matrix"]["gpt-4o"].values()) == {"required"}
    assert "gpt-4o" in reviewed


def test_manifest_estimands_do_not_overclaim_latent_constructs():
    manifest = json.loads((PROJECT_ROOT / "config/experiments.json").read_text())
    estimands = {
        item["id"]: item["estimand"].lower() for item in manifest["experiments"]
    }
    assert "generosity" not in estimands["dictator"]
    assert "fairness" not in estimands["ultimatum"]
    assert "reciprocity" not in estimands["trust_game"]
    assert "iterated" not in estimands["beauty_contest"]
