"""Regression checks for findings from the first independent audit."""

import re
from collections import Counter, defaultdict

import pytest

from src.results.aggregation import (
    _diagnostic_rate,
    _isotonic_majority_threshold,
    _present_bias_pattern,
    _wilson_interval,
)
from src.tasks import engine
from src.tasks.config import experiment_config
from src.tasks.independence import displayed_percentages
from src.tasks.specs import bisection_conditions, fixed_trial_plans


def test_every_displayed_probability_vector_sums_to_one_hundred():
    config = experiment_config("independence")
    for condition in bisection_conditions(config, "audit-seed"):
        probabilities = (
            condition["reference_p_high"],
            condition["reference_p_middle"],
            condition["reference_p_low"],
        )
        assert sum(displayed_percentages(probabilities)) == pytest.approx(100)
    for numerator in range(1025):
        probability = numerator / 1024
        assert sum(displayed_percentages((probability, 1 - probability))) == pytest.approx(100)


def test_rendered_independence_options_sum_to_one_hundred():
    from src.tasks.independence import MMTrianglePrompts, TrianglePoint

    prompt = MMTrianglePrompts.binary_choice(
        TrianglePoint(1 / 12, 1 / 12), 341 / 1024, "Y"
    )
    option_lines = [line for line in prompt.splitlines() if line.startswith("Option")]
    for line in option_lines:
        percentages = [float(value) for value in re.findall(r"([0-9.]+)%", line)]
        assert sum(percentages) == pytest.approx(100)


def test_stag_labels_are_balanced_and_semantically_recorded():
    plans = fixed_trial_plans(experiment_config("stag_hunt"), "audit-seed")
    labels = defaultdict(Counter)
    for plan in plans:
        labels[plan.condition_id][plan.condition["safe_action_label"]] += 1
        assert plan.condition["payoff_dominant_action_label"] != plan.condition["safe_action_label"]
    assert all(counts == {"A": 5, "B": 5} for counts in labels.values())


def test_matching_design_balances_roles_and_choice_order():
    plans = fixed_trial_plans(
        experiment_config("matching_pennies"), "audit-seed"
    )
    role_counts = Counter(plan.role for plan in plans)
    assert role_counts == {"matching": 300, "mismatching": 300}
    by_condition = defaultdict(Counter)
    for plan in plans:
        by_condition[plan.condition_id][plan.condition["choice_order"]] += 1
    assert all(
        counts == {"heads_first": 50, "tails_first": 50}
        for counts in by_condition.values()
    )


def test_seeded_order_is_deterministic_and_permuted():
    config = experiment_config("dictator")
    manifest_order = fixed_trial_plans(config)
    first = fixed_trial_plans(config, "audit-seed")
    second = fixed_trial_plans(config, "audit-seed")
    first_ids = [plan.trial_id for plan in first]
    assert first_ids == [plan.trial_id for plan in second]
    assert first_ids != [plan.trial_id for plan in manifest_order]


def test_diagnostics_exclude_invalid_responses_from_the_denominator():
    records = [
        {
            "trial": {
                "validity": {"status": "valid"},
                "condition": {"expected_semantic_choice": "sooner"},
                "trial_metrics": {"semantic_choice": "sooner"},
            }
        },
        {
            "trial": {
                "validity": {"status": "invalid_response"},
                "condition": {"expected_semantic_choice": "sooner"},
                "trial_metrics": {},
            }
        },
    ]
    assert _diagnostic_rate(records) == {
        "checks": 1,
        "passed": 1,
        "pass_rate": 1,
    }


def test_present_bias_direction_and_ultimatum_threshold_are_prespecified():
    assert _present_bias_pattern(20.5, 50.5, 100)
    assert not _present_bias_pattern(50.5, 20.5, 100)
    monotone, threshold, fitted = _isotonic_majority_threshold([
        {"offer_share": 0, "acceptance_rate": 1, "valid_trials": 20},
        {"offer_share": 0.1, "acceptance_rate": 0, "valid_trials": 20},
        {"offer_share": 0.2, "acceptance_rate": 1, "valid_trials": 20},
    ])
    assert monotone is False
    assert fitted == pytest.approx([0.5, 0.5, 1])
    assert threshold == 0


def test_wilson_interval_exposes_small_sample_uncertainty():
    lower, upper = _wilson_interval(5, 10)
    assert lower == pytest.approx(0.2366, abs=0.001)
    assert upper == pytest.approx(0.7634, abs=0.001)


def test_native_batch_rejects_a_dirty_collection_snapshot(monkeypatch):
    def reject(_root):
        raise RuntimeError("native benchmark collection requires a clean Git working tree")

    monkeypatch.setattr(engine, "require_clean_repository", reject)
    with pytest.raises(RuntimeError, match="clean Git working tree"):
        engine.run_batch(
            "gpt-5.2", run_id="dirty-run", experiment_ids=["dictator"]
        )
