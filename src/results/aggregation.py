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


def _wilson_interval(successes: int, trials: int) -> tuple[float | None, float | None]:
    if trials == 0:
        return None, None
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1 + z**2 / trials
    center = (rate + z**2 / (2 * trials)) / denominator
    half_width = z * math.sqrt(
        rate * (1 - rate) / trials + z**2 / (4 * trials**2)
    ) / denominator
    return center - half_width, center + half_width


def _isotonic_majority_threshold(
    curve: list[dict[str, Any]],
) -> tuple[bool, float | None, list[float | None]]:
    """Fit a weighted nondecreasing curve and return its majority threshold."""
    raw_monotone = all(
        left["acceptance_rate"] is not None
        and right["acceptance_rate"] is not None
        and left["acceptance_rate"] <= right["acceptance_rate"] + 1e-12
        for left, right in zip(curve, curve[1:])
    )
    blocks = []
    for index, point in enumerate(curve):
        rate = point["acceptance_rate"]
        weight = point["valid_trials"]
        if rate is None or weight <= 0:
            continue
        blocks.append({
            "indices": [index],
            "weight": weight,
            "weighted_sum": rate * weight,
        })
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = left["weighted_sum"] / left["weight"]
            right_mean = right["weighted_sum"] / right["weight"]
            if left_mean <= right_mean + 1e-12:
                break
            blocks[-2:] = [{
                "indices": left["indices"] + right["indices"],
                "weight": left["weight"] + right["weight"],
                "weighted_sum": left["weighted_sum"] + right["weighted_sum"],
            }]

    fitted: list[float | None] = [None] * len(curve)
    for block in blocks:
        mean = block["weighted_sum"] / block["weight"]
        for index in block["indices"]:
            fitted[index] = mean
    threshold = next(
        (
            point["offer_share"] for point, estimate in zip(curve, fitted)
            if estimate is not None and estimate >= 0.5
        ),
        None,
    )
    return raw_monotone, threshold, fitted


def _present_bias_pattern(
    immediate_indifference: float,
    delayed_indifference: float,
    larger_amount: float,
    minimum_share_difference: float = 0.02,
) -> bool:
    return (
        delayed_indifference - immediate_indifference
        > larger_amount * minimum_share_difference
    )


def _sample(
    records: list[dict[str, Any]], primary_values: list[float] | None = None
) -> dict[str, Any]:
    counts = {
        status: sum(_trial(record)["validity"]["status"] == status for record in records)
        for status in ("valid", "invalid_response", "provider_error", "interrupted")
    }
    observed = len(records)
    valid_rate = counts["valid"] / observed if observed else None
    invalid_rate = counts["invalid_response"] / observed if observed else None
    return {
        "observed_trials": observed,
        "valid_trials": counts["valid"],
        "invalid_response_trials": counts["invalid_response"],
        "provider_error_trials": counts["provider_error"],
        "interrupted_trials": counts["interrupted"],
        "valid_rate": valid_rate,
        "invalid_response_rate": invalid_rate,
        "valid_rate_standard_error": (
            math.sqrt(valid_rate * (1 - valid_rate) / observed)
            if valid_rate is not None else None
        ),
        "invalid_response_rate_standard_error": (
            math.sqrt(invalid_rate * (1 - invalid_rate) / observed)
            if invalid_rate is not None else None
        ),
        "primary_estimate_standard_error": (
            statistics.stdev(primary_values) / math.sqrt(len(primary_values))
            if primary_values is not None and len(primary_values) >= 2 else None
        ),
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
        "sample": _sample(records, [
            _trial(record)["trial_metrics"]["transfer_share"] for record in valid
        ]),
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
        raw_monotone, minimum, isotonic = _isotonic_majority_threshold(curve)
        for point, fitted_rate in zip(curve, isotonic):
            point["isotonic_acceptance_rate"] = fitted_rate
        responder_by_pool.append(
            {
                "condition_id": f"responder-pool-{_condition_number(pool)}",
                "pool_amount": pool,
                "minimum_acceptable_offer_share": minimum,
                "raw_acceptance_curve_monotone": raw_monotone,
                "acceptance_curve": curve,
            }
        )

    return {
        "sample": _sample(records, [
            _trial(record)["trial_metrics"]["offer_share"] for record in proposer
        ]),
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
        "sample": _sample(records, [
            _trial(record)["trial_metrics"]["send_share"] for record in sender
        ]),
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
        "sample": _sample(records, [
            float(_trial(record)["trial_metrics"]["action"] == "stag")
            for record in valid
        ]),
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
        "sample": _sample(records, all_guesses),
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
        "sample": _sample(records, [
            float(_trial(record)["trial_metrics"]["action"] == "pass")
            for record in valid
        ]),
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
        "sample": _sample(records, [
            _trial(record)["trial_metrics"]["contribution_share"] for record in valid
        ]),
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
        "sample": _sample(records, [
            _trial(record)["trial_metrics"]["normalized_claim"] for record in valid
        ]),
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
    conditions = sorted({
        (
            _trial(record)["condition"]["win_payoff"],
            _trial(record)["condition"].get("payoff_role", "matching"),
        )
        for record in records
    })
    rows = []
    for payoff, role in conditions:
        relevant = [
            record
            for record in valid
            if _trial(record)["condition"]["win_payoff"] == payoff
            and _trial(record)["condition"].get("payoff_role", "matching") == role
        ]
        heads_count = sum(
            _trial(record)["trial_metrics"]["choice"] == "heads"
            for record in relevant
        )
        heads = heads_count / len(relevant) if relevant else None
        lower, upper = _wilson_interval(heads_count, len(relevant))
        rows.append(
            {
                "condition_id": f"win-{_condition_number(payoff)}-role-{role}",
                "win_payoff": payoff,
                "payoff_role": role,
                "heads_rate": heads,
                "tails_rate": 1 - heads if heads is not None else None,
                "heads_rate_ci95_lower": lower,
                "heads_rate_ci95_upper": upper,
                "absolute_deviation_from_half": (
                    abs(heads - 0.5) if heads is not None else None
                ),
                "valid_trials": len(relevant),
            }
        )
    overall_heads_count = sum(
        _trial(record)["trial_metrics"]["choice"] == "heads"
        for record in valid
    )
    heads = overall_heads_count / len(valid) if valid else None
    lower, upper = _wilson_interval(overall_heads_count, len(valid))
    return {
        "sample": _sample(records, [
            float(_trial(record)["trial_metrics"]["choice"] == "heads")
            for record in valid
        ]),
        "overall_heads_rate": heads,
        "overall_tails_rate": 1 - heads if heads is not None else None,
        "overall_heads_rate_ci95_lower": lower,
        "overall_heads_rate_ci95_upper": upper,
        "overall_absolute_deviation_from_half": (
            abs(heads - 0.5) if heads is not None else None
        ),
        "by_condition": rows,
    }


def _final_bisection_bounds(
    group: list[dict[str, Any]], experiment_id: str
) -> tuple[float, float]:
    valid = _valid(group)
    if not valid:
        raise ValueError("bisection sequence has no valid trials")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in valid:
        grouped[_trial(record)["condition"]["bisection_iteration"]].append(record)
    final_step = grouped[max(grouped)]
    condition = _trial(final_step[0])["condition"]
    expected = condition.get("bisection_repetitions_per_step", 1)
    if len(final_step) != expected:
        raise ValueError("bisection sequence ends with an incomplete step")
    lower = condition["lower_bound_before"]
    upper = condition["upper_bound_before"]
    midpoint = condition["midpoint"]
    choices = [_trial(record)["trial_metrics"]["semantic_choice"] for record in final_step]
    choice = max(set(choices), key=choices.count)
    if choices.count(choice) <= expected // 2:
        raise ValueError("bisection step has no majority choice")
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


def _bisection_value(
    records: list[dict[str, Any]], experiment_id: str
) -> float | None:
    if not records or len(_valid(records)) != len(records):
        return None
    lower, upper = _final_bisection_bounds(records, experiment_id)
    return (lower + upper) / 2


def _diagnostic_rate(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid_records = _valid(records)
    passed = 0
    for record in valid_records:
        trial = _trial(record)
        expected = trial["condition"]["expected_semantic_choice"]
        actual = trial["trial_metrics"]["semantic_choice"]
        passed += expected == "either" or actual == expected
    checks = len(valid_records)
    return {
        "checks": checks,
        "passed": passed,
        "pass_rate": passed / checks if checks else None,
    }


def _fit_quadratic_utility(
    points: list[dict[str, Any]], *, eu_beta_tolerance: float
) -> dict[str, Any]:
    """Fit a normalized quadratic utility on the probability simplex."""
    empty = {
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
    usable = [
        point for point in points if point["indifference_probability"] is not None
    ]
    if len(usable) < 6:
        return empty

    import numpy as np

    design = []
    targets = []
    for point in usable:
        reference = (
            point["reference_p_middle"], point["reference_p_high"]
        )
        value = point["indifference_probability"]
        axis = (1 - value, value) if point["axis"] == "y" else (1 - value, 0.0)

        def terms(probabilities):
            middle, high = probabilities
            return (
                middle,
                0.5 * middle**2,
                middle * high,
            )

        reference_terms = terms(reference)
        axis_terms = terms(axis)
        design.append([
            left - right for left, right in zip(reference_terms, axis_terms)
        ])
        targets.append(-(reference[1] - axis[1]))

    matrix = np.asarray(design, dtype=float)
    target = np.asarray(targets, dtype=float)
    try:
        parameters, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    except Exception:
        return {**empty, "status": "failed"}
    if rank < matrix.shape[1]:
        return empty
    alpha_middle, beta_middle_middle, beta_middle_high = (
        float(value) for value in parameters
    )
    predictions = matrix @ parameters
    residual_loss = float(np.mean((target - predictions) ** 2))
    beta_norm = float(np.linalg.norm([
        beta_middle_middle, beta_middle_high
    ]))
    return {
        "status": "success",
        "alpha_low": 0.0,
        "alpha_middle": alpha_middle,
        "alpha_high": 1.0,
        "beta_low_low": 0.0,
        "beta_middle_middle": beta_middle_middle,
        "beta_high_high": 0.0,
        "beta_low_middle": 0.0,
        "beta_low_high": 0.0,
        "beta_middle_high": beta_middle_high,
        "beta_norm": beta_norm,
        "expected_utility_consistent": beta_norm <= eu_beta_tolerance,
        "residual_loss": residual_loss,
    }


def _independence(records: list[dict[str, Any]]) -> dict[str, Any]:
    primary_records = [
        record for record in records
        if _trial(record)["condition"].get("phase") is None
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in primary_records:
        groups[_trial(record)["condition_id"]].append(record)
    points = []
    for condition_id, group in sorted(groups.items()):
        valid = _valid(group)
        source = _trial(group[0])["condition"]
        if not valid or len(valid) != len(group):
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

    parameters = records[0]["metadata"]["experiment"]["parameters"]
    quadratic = _fit_quadratic_utility(
        points,
        eu_beta_tolerance=parameters["quadratic_eu_beta_norm_tolerance"],
    )
    validation_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bidirectional_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    monotonicity = []
    transitivity = []
    for record in records:
        trial = _trial(record)
        phase = trial["condition"].get("phase")
        if phase == "validation":
            validation_groups[trial["condition_id"]].append(record)
        elif phase == "diagnostic_bidirectional":
            bidirectional_groups[trial["condition_id"]].append(record)
        elif phase == "diagnostic_monotonicity":
            monotonicity.append(record)
        elif phase == "diagnostic_transitivity":
            transitivity.append(record)

    validation_deviations = []
    for group in validation_groups.values():
        source_id = _trial(group[0])["condition"]["source_condition_id"]
        original = _bisection_value(groups.get(source_id, []), "independence")
        retest = _bisection_value(group, "independence")
        if original is not None and retest is not None:
            validation_deviations.append(abs(original - retest))

    bidirectional_deviations = []
    for group in bidirectional_groups.values():
        source_id = _trial(group[0])["condition"]["source_condition_id"]
        original = _bisection_value(groups.get(source_id, []), "independence")
        swapped = _bisection_value(group, "independence")
        if original is not None and swapped is not None:
            bidirectional_deviations.append(abs(original - swapped))

    monotonicity_rate = _diagnostic_rate(monotonicity)
    transitivity_rate = _diagnostic_rate(transitivity)
    mean_bidirectional = _mean(bidirectional_deviations)
    positional_bias = (
        mean_bidirectional > 0.1 if mean_bidirectional is not None else None
    )
    scores = [
        rate["pass_rate"] for rate in (monotonicity_rate, transitivity_rate)
        if rate["pass_rate"] is not None
    ]
    rationality_score = None
    if scores:
        rationality_score = max(
            0.0, min(1.0, statistics.fmean(scores) - (0.2 if positional_bias else 0))
        )
    return {
        "sample": _sample(records),
        "indifference_points": points,
        "parallelism": parallelism,
        "quadratic_utility": quadratic,
        "validation": {
            "retests": len(validation_deviations),
            "mean_absolute_deviation": _mean(validation_deviations),
            "maximum_absolute_deviation": (
                max(validation_deviations) if validation_deviations else None
            ),
        },
        "diagnostics": {
            "monotonicity": monotonicity_rate,
            "transitivity": transitivity_rate,
            "bidirectional": {
                "samples": len(bidirectional_deviations),
                "mean_absolute_difference": mean_bidirectional,
                "positional_bias_detected": positional_bias,
            },
            "rationality_score": rationality_score,
        },
    }


def _time(records: list[dict[str, Any]]) -> dict[str, Any]:
    primary_records = [
        record for record in records
        if _trial(record)["condition"].get("phase") is None
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in primary_records:
        groups[_trial(record)["condition_id"]].append(record)
    estimates = []
    for condition_id, group in sorted(groups.items()):
        source = _trial(group[0])["condition"]
        if len(_valid(group)) == len(group):
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
    best = min(successful, key=lambda fit: fit["bic"])["model"] if successful else None

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
            bias_tests.append(_present_bias_pattern(
                immediate["indifference_amount"],
                estimate["indifference_amount"],
                estimate["larger_amount"],
                records[0]["metadata"]["experiment"]["parameters"][
                    "present_bias_minimum_share_difference"
                ],
            ))
    validation_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bidirectional_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    monotonicity = []
    for record in records:
        trial = _trial(record)
        phase = trial["condition"].get("phase")
        if phase == "validation":
            validation_groups[trial["condition_id"]].append(record)
        elif phase == "diagnostic_bidirectional":
            bidirectional_groups[trial["condition_id"]].append(record)
        elif phase == "diagnostic_monotonicity":
            monotonicity.append(record)

    validation_deviations = []
    for group in validation_groups.values():
        source_id = _trial(group[0])["condition"]["source_condition_id"]
        original = _bisection_value(groups.get(source_id, []), "time")
        retest = _bisection_value(group, "time")
        if original is not None and retest is not None:
            validation_deviations.append(abs(original - retest))

    bidirectional_deviations = []
    for group in bidirectional_groups.values():
        source_id = _trial(group[0])["condition"]["source_condition_id"]
        original = _bisection_value(groups.get(source_id, []), "time")
        swapped = _bisection_value(group, "time")
        if original is not None and swapped is not None:
            bidirectional_deviations.append(abs(original - swapped))

    return {
        "sample": _sample(records),
        "discount_estimates": estimates,
        "model_fits": fits,
        "best_fit": best,
        "best_fit_criterion": "bic",
        "validation": {
            "retests": len(validation_deviations),
            "mean_absolute_deviation": _mean(validation_deviations),
        },
        "diagnostics": {
            "monotonicity": _diagnostic_rate(monotonicity),
            "present_bias": {
                "tests": len(bias_tests),
                "biased": sum(bias_tests),
                "bias_rate": sum(bias_tests) / len(bias_tests) if bias_tests else None,
                "minimum_share_difference": records[0]["metadata"]["experiment"][
                    "parameters"
                ]["present_bias_minimum_share_difference"],
            },
            "bidirectional": {
                "samples": len(bidirectional_deviations),
                "mean_absolute_difference": _mean(bidirectional_deviations),
            },
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
                "bic": None,
                "parameter_count": 1 if model != "quasi_hyperbolic" else 2,
                "observations": len(estimates),
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
            [0.95], 1,
            (0.01, 1.0),
        ),
        (
            "hyperbolic",
            lambda time, k: 1 / (1 + k * time),
            [0.01], 1,
            (0.0001, 1.0),
        ),
        (
            "quasi_hyperbolic",
            lambda time, beta, delta: beta * (delta ** (time / 365)),
            [0.9, 0.95], 2,
            ([0.01, 0.01], [1.0, 1.0]),
        ),
    ]
    for model, function, initial, parameter_count, bounds in definitions:
        try:
            parameters, _ = curve_fit(
                function, delays, factors, p0=initial, bounds=bounds, maxfev=10000
            )
            predictions = function(delays, *parameters)
            sse = float(np.sum((factors - predictions) ** 2))
            mean_squared_error = max(sse / len(factors), 1e-15)
            bic = float(
                len(factors) * math.log(mean_squared_error)
                + parameter_count * math.log(len(factors))
            )
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
                    "bic": bic,
                    "parameter_count": parameter_count,
                    "observations": len(factors),
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
                    "bic": None,
                    "parameter_count": parameter_count,
                    "observations": len(factors),
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
