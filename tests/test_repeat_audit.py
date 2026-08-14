"""Regression checks for the independent repeat audit."""

from __future__ import annotations

import copy
import json
import math
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from scipy.optimize import brentq

from src.models.anthropic import wrapper as anthropic_wrapper
from src.models.google import wrapper as google_wrapper
from src.models.openai import wrapper as openai_wrapper
from src.results.aggregation import (
    _discount_fits,
    _fit_quadratic_utility,
    _isotonic_majority_threshold,
)
from src.results.io import read_jsonl, write_jsonl
from src.results.validation import validate_record
from src.tasks import engine
from src.tasks.beauty_contest import parse_number as parse_beauty
from src.tasks.config import canonical_run_paths, experiment_config
from src.tasks.dictator import parse_dollar_amount as parse_dictator
from src.tasks.public_goods import parse_contribution
from src.tasks.response_formats import parse_bounded_amount
from src.tasks.specs import (
    bisection_conditions,
    bisection_plan,
    bidirectional_bisection_conditions,
    validation_bisection_conditions,
)
from src.tasks.travellers_dilemma import parse_number as parse_traveller
from src.tasks.trust_game import parse_dollar_amount as parse_trust
from src.tasks.ultimatum import parse_dollar_amount as parse_ultimatum


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "token",
    ["1,00", "1,2", "1,,000", "1,000,", "12,34,567", "1,0000"],
)
def test_shared_amount_parser_rejects_malformed_grouping(token):
    assert parse_bounded_amount(
        f"AMOUNT={token}", maximum=2_000_000, labels=("amount",)
    ) is None


def test_shared_amount_parser_accepts_only_canonical_grouping_and_cent_precision():
    assert parse_bounded_amount(
        "AMOUNT=1,000.25", maximum=2_000, labels=("amount",)
    ) == 1000.25
    assert parse_bounded_amount(
        "AMOUNT=1000.25", maximum=2_000, labels=("amount",)
    ) == 1000.25
    assert parse_bounded_amount(
        "AMOUNT=1,000.251", maximum=2_000, labels=("amount",)
    ) is None


@pytest.mark.parametrize(
    ("parser", "response"),
    [
        (lambda value: parse_dictator(value, 1000), "TRANSFER=1,0"),
        (lambda value: parse_ultimatum(value, 1000), "OFFER=1,00"),
        (lambda value: parse_trust(value, 1000, "send"), "SEND=1,2"),
        (lambda value: parse_trust(value, 3000, "return"), "RETURN=2,5"),
        (lambda value: parse_beauty(value, 0, 100), "CHOICE=1,0"),
        (lambda value: parse_contribution(value, 1000), "CONTRIBUTION=1,00"),
        (lambda value: parse_traveller(value, 20, 1000, 10), "CLAIM=1,0"),
    ],
)
def test_every_amount_parser_rejects_ambiguous_grouping(parser, response):
    assert parser(response) is None


def test_validation_retests_preserve_order_and_bidirectional_checks_reverse_it():
    for experiment_id in ("independence", "time"):
        config = experiment_config(experiment_id)
        seed = "repeat-audit"
        primary = {
            condition["condition_id"]: condition
            for condition in bisection_conditions(config, seed)
        }
        for condition in validation_bisection_conditions(config, seed):
            source = primary[condition["source_condition_id"]]
            assert condition["swap_order"] == source["swap_order"]
        for condition in bidirectional_bisection_conditions(config, seed):
            source = primary[condition["source_condition_id"]]
            assert condition["swap_order"] != source["swap_order"]


def test_bisection_advances_with_three_response_majority():
    config = experiment_config("time")
    config["settings"]["bisection_iterations"] = 2
    base = bisection_conditions(config, "majority-test")[0]
    trials = []
    first_midpoint = None
    for semantic_choice in ("sooner", "later", "sooner"):
        plan = bisection_plan(config, base, trials)
        first_midpoint = first_midpoint or plan.condition["midpoint"]
        assert plan.condition["midpoint"] == first_midpoint
        trials.append({
            "condition": plan.condition,
            "trial_metrics": {"semantic_choice": semantic_choice},
            "validity": {"status": "valid"},
        })
    next_plan = bisection_plan(config, base, trials)
    assert next_plan.condition["bisection_iteration"] == 2
    assert next_plan.condition["midpoint"] == pytest.approx(first_midpoint / 2)


def _quadratic_points(alpha_middle, beta_middle_middle, beta_middle_high):
    def utility(middle, high):
        return (
            alpha_middle * middle
            + high
            + 0.5 * beta_middle_middle * middle**2
            + beta_middle_high * middle * high
        )

    points = []
    for low in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        for high in (0.1, 0.2, 0.3):
            if low + high >= 0.95:
                continue
            middle = 1 - low - high
            target = utility(middle, high)
            axis = "y" if target >= utility(1, 0) else "x"
            if axis == "y":
                function = lambda value: utility(1 - value, value) - target
            else:
                function = lambda value: utility(1 - value, 0) - target
            value = brentq(function, 0, 1)
            points.append({
                "axis": axis,
                "reference_p_low": low,
                "reference_p_middle": middle,
                "reference_p_high": high,
                "indifference_probability": value,
            })
    return points


def test_quadratic_utility_fit_recovers_known_preferences():
    fit = _fit_quadratic_utility(
        _quadratic_points(0.55, 0.18, 0.08), eu_beta_tolerance=0.05
    )
    assert fit["status"] == "success"
    assert fit["alpha_middle"] == pytest.approx(0.55, abs=1e-8)
    assert fit["beta_middle_middle"] == pytest.approx(0.18, abs=1e-8)
    assert fit["beta_middle_high"] == pytest.approx(0.08, abs=1e-8)
    assert fit["expected_utility_consistent"] is False

    expected_utility = _fit_quadratic_utility(
        _quadratic_points(0.5, 0, 0), eu_beta_tolerance=0.05
    )
    assert expected_utility["status"] == "success"
    assert expected_utility["expected_utility_consistent"] is True


@pytest.mark.parametrize(
    ("expected", "factor"),
    [
        ("exponential", lambda delay: 0.9 ** (delay / 365)),
        ("hyperbolic", lambda delay: 1 / (1 + 0.002 * delay)),
        (
            "quasi_hyperbolic",
            lambda delay: 0.78 * (0.92 ** (delay / 365)),
        ),
    ],
)
def test_bic_recovers_prespecified_discount_models(expected, factor):
    estimates = [
        {"delay_days": delay, "discount_factor": factor(delay)}
        for delay in (30, 90, 180, 365, 730, 1095, 1460, 1825)
    ]
    successful = [
        fit for fit in _discount_fits(estimates) if fit["status"] == "success"
    ]
    assert min(successful, key=lambda fit: fit["bic"])["model"] == expected


def test_isotonic_threshold_recovers_a_logistic_majority_boundary():
    generator = random.Random(20260812)
    recovered = 0
    for _ in range(500):
        curve = []
        for index in range(21):
            share = index / 20
            probability = 1 / (1 + math.exp(-20 * (share - 0.3)))
            accepted = sum(generator.random() < probability for _ in range(20))
            curve.append({
                "offer_share": share,
                "acceptance_rate": accepted / 20,
                "valid_trials": 20,
            })
        _, threshold, _ = _isotonic_majority_threshold(curve)
        recovered += threshold is not None and abs(threshold - 0.3) <= 0.1 + 1e-12
    assert recovered / 500 >= 0.95


def test_provider_clients_disable_nested_sdk_retries(monkeypatch):
    openai_calls = []
    monkeypatch.setattr(
        openai_wrapper,
        "OpenAI",
        lambda **kwargs: openai_calls.append(kwargs) or SimpleNamespace(),
    )
    openai_wrapper.LLMInterface("gpt-4o-2024-11-20")
    assert openai_calls[0]["max_retries"] == 0

    anthropic_calls = []
    monkeypatch.setattr(
        anthropic_wrapper,
        "anthropic",
        SimpleNamespace(
            Anthropic=lambda **kwargs: anthropic_calls.append(kwargs)
            or SimpleNamespace()
        ),
    )
    anthropic_wrapper.LLMInterface("claude-haiku-4-5-20251001")
    assert anthropic_calls[0]["max_retries"] == 0

    google_calls = []
    monkeypatch.setattr(
        google_wrapper,
        "genai",
        SimpleNamespace(
            Client=lambda **kwargs: google_calls.append(kwargs) or SimpleNamespace()
        ),
    )
    google_wrapper.LLMInterface("gemini-2.5-flash")
    retry_options = google_calls[0]["http_options"].retry_options
    assert retry_options.attempts == 1


def test_recorded_outer_attempts_equal_wrapper_transport_calls(monkeypatch):
    calls = 0

    class Completions:
        def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            raise TimeoutError("transient")

    interface = object.__new__(openai_wrapper.LLMInterface)
    interface.model_id = "gpt-4o-2024-11-20"
    interface._api_mode = "chat"
    interface.client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    monkeypatch.setattr(openai_wrapper, "log_model_call", lambda **_kwargs: None)
    completion = engine.request_model_completion(
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
    assert completion.attempts == calls == 3


def _example_trial():
    path = PROJECT_ROOT / "schemas" / "examples" / "result-record.trial.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_native_validation_rejects_dirty_and_stale_manifest_metadata():
    dirty = _example_trial()
    dirty["metadata"]["provenance"]["repository_dirty"] = True
    assert "dirty_native_provenance" in {
        finding.code for finding in validate_record(dirty)
    }

    stale = _example_trial()
    stale["metadata"]["experiment"]["parameters"]["pool_amounts"] = [10]
    assert "experiment_manifest_mismatch" in {
        finding.code for finding in validate_record(stale)
    }


@pytest.mark.parametrize("stale_field", ["revision", "parameters"])
def test_resume_rejects_stale_metadata_before_provider_access(
    tmp_path, stale_field
):
    run_id = f"stale-{stale_field}"
    engine.run_experiment(
        "gpt-4o",
        "dictator",
        run_id=run_id,
        interface=engine.FixtureModel(),
        release_root=tmp_path,
        sleeper=lambda _seconds: None,
    )
    paths = canonical_run_paths(
        "gpt-4o", "dictator", run_id, release_root=tmp_path
    )
    records = read_jsonl(paths.raw)
    for record in records:
        if stale_field == "revision":
            record["metadata"]["provenance"]["code_revision"] = "0" * 40
        else:
            record["metadata"]["experiment"]["parameters"]["pool_amounts"] = [10]
    write_jsonl(paths.raw, records)

    class MustNotRun:
        def generate_response(self, **_kwargs):
            raise AssertionError("provider access preceded resume validation")

    with pytest.raises(ValueError, match="stale"):
        engine.run_experiment(
            "gpt-4o",
            "dictator",
            run_id=run_id,
            interface=MustNotRun(),
            resume=True,
            release_root=tmp_path,
            sleeper=lambda _seconds: None,
        )


def test_default_checkpoints_preserve_a_clean_isolated_repository(tmp_path):
    (tmp_path / ".gitignore").write_text("data/releases/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=EconBench", "-c",
            "user.email=econbench@example.invalid", "commit", "-qm", "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    for experiment_id in ("dictator", "beauty_contest"):
        result = engine.run_experiment(
            "gpt-4o",
            experiment_id,
            run_id=f"clean-{experiment_id}",
            interface=engine.FixtureModel(),
            project_root=tmp_path,
            sleeper=lambda _seconds: None,
        )
        assert result["derived"]["metadata"]["provenance"]["repository_dirty"] is False
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""


def test_public_documentation_uses_the_canonical_runner():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "python src/tasks/" not in readme
    assert "dynamic consistency" not in readme.lower()
    assert "python scripts/run_benchmark.py" in readme


def test_direct_requirements_are_exactly_pinned():
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert requirements
    assert all("==" in requirement for requirement in requirements if requirement)
