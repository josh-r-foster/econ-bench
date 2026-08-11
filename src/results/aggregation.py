"""Aggregate canonical raw trials into experiment metric objects."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Callable


def _trial(record: dict[str, Any]) -> dict[str, Any]:
    return record["trial"]


def _valid(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if _trial(record)["validity"]["status"] == "valid"]


def _mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _sample(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        status: sum(_trial(record)["validity"]["status"] == status for record in records)
        for status in ("valid", "invalid_response", "provider_error", "interrupted")
    }
    observed = len(records)
    return {
        "observed_trials": observed,
        "valid_trials": counts["valid"],
        "invalid_response_trials": counts["invalid_response"],
        "provider_error_trials": counts["provider_error"],
        "interrupted_trials": counts["interrupted"],
        "valid_rate": counts["valid"] / observed if observed else None,
        "invalid_response_rate": counts["invalid_response"] / observed if observed else None,
    }


def _condition_number(value: float) -> str:
    return f"{value:g}"


def _dictator(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = _valid(records)
    pools = sorted({_trial(record)["condition"]["pool_amount"] for record in records})
    by_pool = []
    for pool in pools:
        relevant = [
            record for record in valid if _trial(record)["condition"]["pool_amount"] == pool
        ]
        by_pool.append(
            {
                "condition_id": f"pool-{_condition_number(pool)}",
                "pool_amount": pool,
                "mean_transfer_amount": _mean(
                    [_trial(record)["trial_metrics"]["transfer_amount"] for record in relevant]
                ),
                "mean_transfer_share": _mean(
                    [_trial(record)["trial_metrics"]["transfer_share"] for record in relevant]
                ),
                "valid_trials": len(relevant),
            }
        )
    return {
        "sample": _sample(records),
        "overall_mean_transfer_share": _mean(
            [_trial(record)["trial_metrics"]["transfer_share"] for record in valid]
        ),
        "by_pool": by_pool,
    }


def _ultimatum(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = _valid(records)
    proposer = [record for record in valid if _trial(record)["role"] == "proposer"]
    responder = [record for record in valid if _trial(record)["role"] == "responder"]
    pools = sorted({_trial(record)["condition"]["pool_amount"] for record in records})

    proposer_by_pool = []
    responder_by_pool = []
    for pool in pools:
        pool_proposer = [
            record
            for record in proposer
            if _trial(record)["condition"]["pool_amount"] == pool
        ]
        proposer_by_pool.append(
            {
                "condition_id": f"proposer-pool-{_condition_number(pool)}",
                "pool_amount": pool,
                "mean_offer_amount": _mean(
                    [_trial(record)["trial_metrics"]["offer_amount"] for record in pool_proposer]
                ),
                "mean_offer_share": _mean(
                    [_trial(record)["trial_metrics"]["offer_share"] for record in pool_proposer]
                ),
                "valid_trials": len(pool_proposer),
            }
        )

        pool_responder = [
            record
            for record in responder
            if _trial(record)["condition"]["pool_amount"] == pool
        ]
        offer_shares = sorted(
            {_trial(record)["condition"]["offer_share"] for record in pool_responder}
        )
        curve = []
        for offer_share in offer_shares:
            relevant = [
                record
                for record in pool_responder
                if _trial(record)["condition"]["offer_share"] == offer_share
            ]
            curve.append(
                {
                    "offer_share": offer_share,
                    "acceptance_rate": _mean(
                        [
                            float(_trial(record)["trial_metrics"]["accepted"])
                            for record in relevant
                        ]
                    ),
                    "valid_trials": len(relevant),
                }
            )
        minimum = next(
            (
                point["offer_share"]
                for point in curve
                if point["acceptance_rate"] is not None
                and point["acceptance_rate"] > 0.5
            ),
            None,
        )
        responder_by_pool.append(
            {
                "condition_id": f"responder-pool-{_condition_number(pool)}",
                "pool_amount": pool,
                "minimum_acceptable_offer_share": minimum,
                "acceptance_curve": curve,
            }
        )

    return {
        "sample": _sample(records),
        "overall_mean_offer_share": _mean(
            [_trial(record)["trial_metrics"]["offer_share"] for record in proposer]
        ),
        "proposer_by_pool": proposer_by_pool,
        "responder_by_pool": responder_by_pool,
    }


def _trust_game(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = _valid(records)
    sender = [record for record in valid if _trial(record)["role"] == "sender"]
    receiver = [record for record in valid if _trial(record)["role"] == "receiver"]
    endowments = sorted({_trial(record)["condition"]["endowment"] for record in records})

    sender_rows = []
    for endowment in endowments:
        relevant = [
            record
            for record in sender
            if _trial(record)["condition"]["endowment"] == endowment
        ]
        sender_rows.append(
            {
                "condition_id": f"sender-endowment-{_condition_number(endowment)}",
                "endowment": endowment,
                "mean_amount_sent": _mean(
                    [_trial(record)["trial_metrics"]["amount_sent"] for record in relevant]
                ),
                "mean_send_share": _mean(
                    [_trial(record)["trial_metrics"]["send_share"] for record in relevant]
                ),
                "valid_trials": len(relevant),
            }
        )

    receiver_groups: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for record in receiver:
        condition = _trial(record)["condition"]
        receiver_groups[(condition["endowment"], condition["sent_share"])].append(record)
    receiver_rows = []
    for (endowment, sent_share), relevant in sorted(receiver_groups.items()):
        receiver_rows.append(
            {
                "condition_id": (
                    f"receiver-endowment-{_condition_number(endowment)}-sent-"
                    f"{_condition_number(sent_share)}"
                ),
                "endowment": endowment,
                "sent_share": sent_share,
                "mean_amount_returned": _mean(
                    [_trial(record)["trial_metrics"]["amount_returned"] for record in relevant]
                ),
                "mean_return_share_of_received": _mean(
                    [
                        value
                        for record in relevant
                        if (
                            value := _trial(record)["trial_metrics"][
                                "return_share_of_received"
                            ]
                        )
                        is not None
                    ]
                ),
                "mean_return_multiple_of_sent": _mean(
                    [
                        value
                        for record in relevant
                        if (
                            value := _trial(record)["trial_metrics"][
                                "return_multiple_of_sent"
                            ]
                        )
                        is not None
                    ]
                ),
                "valid_trials": len(relevant),
            }
        )

    return {
        "sample": _sample(records),
        "overall_mean_send_share": _mean(
            [_trial(record)["trial_metrics"]["send_share"] for record in sender]
        ),
        "overall_mean_return_share_of_received": _mean(
            [
                value
                for record in receiver
                if (value := _trial(record)["trial_metrics"]["return_share_of_received"])
                is not None
            ]
        ),
        "sender_by_endowment": sender_rows,
        "receiver_by_condition": receiver_rows,
    }


def _stag_hunt(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = _valid(records)
    groups: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        condition = _trial(record)["condition"]
        groups[(condition["coordination_payoff"], condition["safe_payoff_multiplier"])].append(
            record
        )
    rows = []
    for (payoff, multiplier), group in sorted(groups.items()):
        relevant = _valid(group)
        rows.append(
            {
                "condition_id": (
                    f"payoff-{_condition_number(payoff)}-safe-{_condition_number(multiplier)}"
                ),
                "coordination_payoff": payoff,
                "safe_payoff_multiplier": multiplier,
                "stag_rate": _mean(
                    [
                        float(_trial(record)["trial_metrics"]["action"] == "stag")
                        for record in relevant
                    ]
                ),
                "valid_trials": len(relevant),
            }
        )
    return {
        "sample": _sample(records),
        "overall_stag_rate": _mean(
            [
                float(_trial(record)["trial_metrics"]["action"] == "stag")
                for record in valid
            ]
        ),
        "by_condition": rows,
    }


def _beauty_contest(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = _valid(records)
    prizes = sorted({_trial(record)["condition"]["prize"] for record in records})
    rows = []
    for prize in prizes:
        relevant = [
            record for record in valid if _trial(record)["condition"]["prize"] == prize
        ]
        guesses = [_trial(record)["trial_metrics"]["guess"] for record in relevant]
        rows.append(
            {
                "condition_id": f"prize-{_condition_number(prize)}",
                "prize": prize,
                "mean_guess": _mean(guesses),
                "median_guess": _median(guesses),
                "minimum_guess": min(guesses) if guesses else None,
                "maximum_guess": max(guesses) if guesses else None,
                "valid_trials": len(relevant),
            }
        )
    all_guesses = [_trial(record)["trial_metrics"]["guess"] for record in valid]
    return {
        "sample": _sample(records),
        "overall_mean_guess": _mean(all_guesses),
        "overall_median_guess": _median(all_guesses),
        "by_prize": rows,
    }


def _centipede_game(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = _valid(records)
    groups: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        condition = _trial(record)["condition"]
        groups[(condition["final_payoff_level"], condition["turn"])].append(record)
    rows = []
    for (level, turn), group in sorted(groups.items()):
        relevant = _valid(group)
        pass_rate = _mean(
            [
                float(_trial(record)["trial_metrics"]["action"] == "pass")
                for record in relevant
            ]
        )
        rows.append(
            {
                "condition_id": f"level-{_condition_number(level)}-turn-{turn}",
                "final_payoff_level": level,
                "turn": turn,
                "pass_rate": pass_rate,
                "take_rate": 1 - pass_rate if pass_rate is not None else None,
                "valid_trials": len(relevant),
            }
        )
    pass_rate = _mean(
        [
            float(_trial(record)["trial_metrics"]["action"] == "pass")
            for record in valid
        ]
    )
    take_rate = 1 - pass_rate if pass_rate is not None else None
    return {
        "sample": _sample(records),
        "overall_pass_rate": pass_rate,
        "overall_take_rate": take_rate,
        "backward_induction_consistency_rate": take_rate,
        "by_condition": rows,
    }


def _public_goods(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = _valid(records)
    groups: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        condition = _trial(record)["condition"]
        groups[(condition["endowment"], condition["multiplier"])].append(record)
    rows = []
    for (endowment, multiplier), group in sorted(groups.items()):
        relevant = _valid(group)
        rows.append(
            {
                "condition_id": (
                    f"endowment-{_condition_number(endowment)}-multiplier-"
                    f"{_condition_number(multiplier)}"
                ),
                "endowment": endowment,
                "multiplier": multiplier,
                "mean_contribution_amount": _mean(
                    [
                        _trial(record)["trial_metrics"]["contribution_amount"]
                        for record in relevant
                    ]
                ),
                "mean_contribution_share": _mean(
                    [
                        _trial(record)["trial_metrics"]["contribution_share"]
                        for record in relevant
                    ]
                ),
                "valid_trials": len(relevant),
            }
        )
    return {
        "sample": _sample(records),
        "overall_mean_contribution_share": _mean(
            [_trial(record)["trial_metrics"]["contribution_share"] for record in valid]
        ),
        "by_condition": rows,
    }


def _travellers_dilemma(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = _valid(records)
    levels = sorted({_trial(record)["condition"]["upper_bound_level"] for record in records})
    rows = []
    for level in levels:
        relevant = [
            record
            for record in valid
            if _trial(record)["condition"]["upper_bound_level"] == level
        ]
        source = next(
            _trial(record)["condition"]
            for record in records
            if _trial(record)["condition"]["upper_bound_level"] == level
        )
        rows.append(
            {
                "condition_id": f"upper-{_condition_number(level)}",
                "upper_bound_level": level,
                "lower_bound": source["lower_bound"],
                "upper_bound": source["upper_bound"],
                "bonus": source["bonus"],
                "mean_claim_amount": _mean(
                    [_trial(record)["trial_metrics"]["claim_amount"] for record in relevant]
                ),
                "median_claim_amount": _median(
                    [_trial(record)["trial_metrics"]["claim_amount"] for record in relevant]
                ),
                "mean_normalized_claim": _mean(
                    [
                        _trial(record)["trial_metrics"]["normalized_claim"]
                        for record in relevant
                    ]
                ),
                "mean_claim_on_2_100_scale": _mean(
                    [
                        _trial(record)["trial_metrics"]["claim_on_2_100_scale"]
                        for record in relevant
                    ]
                ),
                "lower_bound_choice_rate": _mean(
                    [
                        float(_trial(record)["trial_metrics"]["lower_bound_choice"])
                        for record in relevant
                    ]
                ),
                "valid_trials": len(relevant),
            }
        )
    return {
        "sample": _sample(records),
        "overall_mean_claim_amount": _mean(
            [_trial(record)["trial_metrics"]["claim_amount"] for record in valid]
        ),
        "overall_mean_normalized_claim": _mean(
            [_trial(record)["trial_metrics"]["normalized_claim"] for record in valid]
        ),
        "overall_mean_claim_on_2_100_scale": _mean(
            [_trial(record)["trial_metrics"]["claim_on_2_100_scale"] for record in valid]
        ),
        "overall_lower_bound_choice_rate": _mean(
            [
                float(_trial(record)["trial_metrics"]["lower_bound_choice"])
                for record in valid
            ]
        ),
        "by_condition": rows,
    }


def _matching_pennies(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = _valid(records)
    payoffs = sorted({_trial(record)["condition"]["win_payoff"] for record in records})
    rows = []
    for payoff in payoffs:
        relevant = [
            record
            for record in valid
            if _trial(record)["condition"]["win_payoff"] == payoff
        ]
        heads = _mean(
            [
                float(_trial(record)["trial_metrics"]["choice"] == "heads")
                for record in relevant
            ]
        )
        rows.append(
            {
                "condition_id": f"win-{_condition_number(payoff)}",
                "win_payoff": payoff,
                "heads_rate": heads,
                "tails_rate": 1 - heads if heads is not None else None,
                "distance_from_mixed_equilibrium": (
                    abs(heads - 0.5) if heads is not None else None
                ),
                "valid_trials": len(relevant),
            }
        )
    heads = _mean(
        [
            float(_trial(record)["trial_metrics"]["choice"] == "heads")
            for record in valid
        ]
    )
    return {
        "sample": _sample(records),
        "overall_heads_rate": heads,
        "overall_tails_rate": 1 - heads if heads is not None else None,
        "overall_distance_from_mixed_equilibrium": (
            abs(heads - 0.5) if heads is not None else None
        ),
        "by_win_payoff": rows,
    }


def _final_bisection_bounds(
    group: list[dict[str, Any]], experiment_id: str
) -> tuple[float, float]:
    ordered = sorted(
        _valid(group), key=lambda record: _trial(record)["condition"]["bisection_iteration"]
    )
    if not ordered:
        raise ValueError("bisection sequence has no valid trials")
    final = _trial(ordered[-1])
    condition = final["condition"]
    lower = condition["lower_bound_before"]
    upper = condition["upper_bound_before"]
    midpoint = condition["midpoint"]
    choice = final["trial_metrics"]["semantic_choice"]
    if experiment_id == "time":
        if choice == "sooner":
            upper = midpoint
        else:
            lower = midpoint
    elif condition["axis"] == "y":
        if choice == "reference_lottery":
            lower = midpoint
        else:
            upper = midpoint
    else:
        if choice == "reference_lottery":
            upper = midpoint
        else:
            lower = midpoint
    return lower, upper


def _independence(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_trial(record)["condition_id"]].append(record)
    points = []
    for condition_id, group in sorted(groups.items()):
        valid = _valid(group)
        source = _trial(group[0])["condition"]
        if not valid:
            indifference = None
            slope = None
            valid_sequences = 0
        else:
            lower, upper = _final_bisection_bounds(group, "independence")
            indifference = (lower + upper) / 2
            if source["axis"] == "y" and source["reference_p_low"]:
                slope = (indifference - source["reference_p_high"]) / (
                    -source["reference_p_low"]
                )
            elif source["axis"] == "x" and source["reference_p_high"]:
                slope = (-source["reference_p_high"]) / (
                    indifference - source["reference_p_low"]
                )
            else:
                slope = None
            valid_sequences = 1
        points.append(
            {
                "condition_id": condition_id,
                "axis": source["axis"],
                "reference_p_low": source["reference_p_low"],
                "reference_p_middle": source["reference_p_middle"],
                "reference_p_high": source["reference_p_high"],
                "indifference_probability": indifference,
                "local_slope": slope,
                "valid_sequences": valid_sequences,
            }
        )

    slopes = [point["local_slope"] for point in points if point["local_slope"] is not None]
    if len(slopes) < 2:
        parallelism = {
            "status": "insufficient_data",
            "mean_slope": None,
            "slope_standard_deviation": None,
            "coefficient_of_variation": None,
            "slope_position_correlation": None,
            "pattern": "insufficient_data",
            "independence_violated": None,
        }
    else:
        mean_slope = statistics.fmean(slopes)
        slope_sd = statistics.pstdev(slopes)
        coefficient = slope_sd / abs(mean_slope) if mean_slope else None
        pattern = "parallel" if coefficient is not None and coefficient < 0.15 else "irregular"
        parallelism = {
            "status": "success",
            "mean_slope": mean_slope,
            "slope_standard_deviation": slope_sd,
            "coefficient_of_variation": coefficient,
            "slope_position_correlation": None,
            "pattern": pattern,
            "independence_violated": pattern != "parallel",
        }

    quadratic = {
        "status": "insufficient_data",
        "alpha_low": None,
        "alpha_middle": None,
        "alpha_high": None,
        "beta_low_low": None,
        "beta_middle_middle": None,
        "beta_high_high": None,
        "beta_low_middle": None,
        "beta_low_high": None,
        "beta_middle_high": None,
        "beta_norm": None,
        "expected_utility_consistent": None,
        "residual_loss": None,
    }
    empty_rate = {"checks": 0, "passed": 0, "pass_rate": None}
    return {
        "sample": _sample(records),
        "indifference_points": points,
        "parallelism": parallelism,
        "quadratic_utility": quadratic,
        "validation": {
            "retests": 0,
            "mean_absolute_deviation": None,
            "maximum_absolute_deviation": None,
        },
        "diagnostics": {
            "monotonicity": dict(empty_rate),
            "transitivity": dict(empty_rate),
            "bidirectional": {
                "samples": 0,
                "mean_absolute_difference": None,
                "positional_bias_detected": None,
            },
            "rationality_score": None,
        },
    }


def _time(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_trial(record)["condition_id"]].append(record)
    estimates = []
    for condition_id, group in sorted(groups.items()):
        source = _trial(group[0])["condition"]
        if _valid(group):
            lower, upper = _final_bisection_bounds(group, "time")
            indifference = (lower + upper) / 2
            factor = indifference / source["larger_amount"]
            if factor > 0 and source["delay_days"] > 0:
                annual_factor = factor ** (365 / source["delay_days"])
                annual_rate = (1 / annual_factor) - 1
            else:
                annual_rate = None
            valid_sequences = 1
        else:
            indifference = factor = annual_rate = None
            valid_sequences = 0
        estimates.append(
            {
                "condition_id": condition_id,
                "larger_amount": source["larger_amount"],
                "delay_days": source["delay_days"],
                "front_end_delay_days": source["front_end_delay_days"],
                "indifference_amount": indifference,
                "discount_factor": factor,
                "annualized_rate": annual_rate,
                "valid_sequences": valid_sequences,
            }
        )

    baseline = [
        estimate
        for estimate in estimates
        if estimate["front_end_delay_days"] == 0 and estimate["discount_factor"] is not None
    ]
    fits = _discount_fits(baseline)
    successful = [fit for fit in fits if fit["status"] == "success"]
    best = min(successful, key=lambda fit: fit["sse"])["model"] if successful else None

    baseline_index = {
        (estimate["larger_amount"], estimate["delay_days"]): estimate
        for estimate in baseline
    }
    bias_tests = []
    for estimate in estimates:
        if estimate["front_end_delay_days"] <= 0 or estimate["indifference_amount"] is None:
            continue
        immediate = baseline_index.get((estimate["larger_amount"], estimate["delay_days"]))
        if immediate is not None:
            bias_tests.append(
                immediate["indifference_amount"] - estimate["indifference_amount"]
                > estimate["larger_amount"] * 0.02
            )
    empty_rate = {"checks": 0, "passed": 0, "pass_rate": None}
    return {
        "sample": _sample(records),
        "discount_estimates": estimates,
        "model_fits": fits,
        "best_fit": best,
        "validation": {"retests": 0, "mean_absolute_deviation": None},
        "diagnostics": {
            "monotonicity": empty_rate,
            "present_bias": {
                "tests": len(bias_tests),
                "biased": sum(bias_tests),
                "bias_rate": sum(bias_tests) / len(bias_tests) if bias_tests else None,
            },
            "bidirectional": {"samples": 0, "mean_absolute_difference": None},
        },
    }


def _discount_fits(estimates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    models = ("exponential", "hyperbolic", "quasi_hyperbolic")
    if len(estimates) < 3:
        return [
            {
                "model": model,
                "status": "insufficient_data",
                "sse": None,
                "beta": None,
                "delta": None,
                "k": None,
                "annualized_rate": None,
                "predictions": [],
            }
            for model in models
        ]

    import numpy as np
    from scipy.optimize import curve_fit

    delays = np.array([estimate["delay_days"] for estimate in estimates])
    factors = np.array([estimate["discount_factor"] for estimate in estimates])
    fits = []

    definitions = [
        (
            "exponential",
            lambda time, delta: delta ** (time / 365),
            [0.95],
            (0.01, 1.0),
        ),
        (
            "hyperbolic",
            lambda time, k: 1 / (1 + k * time),
            [0.01],
            (0.0001, 1.0),
        ),
        (
            "quasi_hyperbolic",
            lambda time, beta, delta: beta * (delta ** (time / 365)),
            [0.9, 0.95],
            ([0.01, 0.01], [1.0, 1.0]),
        ),
    ]
    for model, function, initial, bounds in definitions:
        try:
            parameters, _ = curve_fit(
                function, delays, factors, p0=initial, bounds=bounds, maxfev=10000
            )
            predictions = function(delays, *parameters)
            sse = float(np.sum((factors - predictions) ** 2))
            beta = delta = k = annualized_rate = None
            if model == "exponential":
                delta = float(parameters[0])
                annualized_rate = (1 / delta) - 1
            elif model == "hyperbolic":
                k = float(parameters[0])
            else:
                beta = float(parameters[0])
                delta = float(parameters[1])
                annualized_rate = (1 / delta) - 1
            fits.append(
                {
                    "model": model,
                    "status": "success",
                    "sse": sse,
                    "beta": beta,
                    "delta": delta,
                    "k": k,
                    "annualized_rate": annualized_rate,
                    "predictions": [float(value) for value in predictions],
                }
            )
        except Exception:
            fits.append(
                {
                    "model": model,
                    "status": "failed",
                    "sse": None,
                    "beta": None,
                    "delta": None,
                    "k": None,
                    "annualized_rate": None,
                    "predictions": [],
                }
            )
    return fits


AGGREGATORS: dict[str, Callable[[list[dict[str, Any]]], dict[str, Any]]] = {
    "independence": _independence,
    "time": _time,
    "dictator": _dictator,
    "ultimatum": _ultimatum,
    "trust_game": _trust_game,
    "stag_hunt": _stag_hunt,
    "beauty_contest": _beauty_contest,
    "centipede_game": _centipede_game,
    "public_goods": _public_goods,
    "travellers_dilemma": _travellers_dilemma,
    "matching_pennies": _matching_pennies,
}


def aggregate_trials(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one homogeneous canonical raw result collection."""
    if not records:
        raise ValueError("cannot aggregate an empty trial collection")
    experiment_ids = {
        record["metadata"]["experiment"]["id"] for record in records
    }
    if len(experiment_ids) != 1:
        raise ValueError("trial collection contains multiple experiments")
    experiment_id = experiment_ids.pop()
    try:
        aggregator = AGGREGATORS[experiment_id]
    except KeyError as error:
        raise ValueError(f"unsupported experiment {experiment_id!r}") from error
    return aggregator(records)
