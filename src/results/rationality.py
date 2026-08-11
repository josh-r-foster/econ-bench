"""Combined rationality projection from canonical Independence and Time results."""

from __future__ import annotations

import statistics
from typing import Any


def _annual_factor(estimate: dict[str, Any]) -> float | None:
    factor = estimate["discount_factor"]
    delay = estimate["delay_days"]
    if factor is None or factor <= 0 or delay <= 0:
        return None
    return factor ** (365 / delay)


def _patience(time_metrics: dict[str, Any]) -> float | None:
    exponential = next(
        (
            fit
            for fit in time_metrics["model_fits"]
            if fit["model"] == "exponential" and fit["status"] == "success"
        ),
        None,
    )
    if exponential is not None:
        return exponential["delta"]
    factors = [
        value
        for estimate in time_metrics["discount_estimates"]
        if estimate["front_end_delay_days"] == 0
        and (value := _annual_factor(estimate)) is not None
    ]
    return float(statistics.fmean(factors)) if factors else None


def _magnitude_penalty(time_metrics: dict[str, Any]) -> float:
    by_amount: dict[float, list[float]] = {}
    for estimate in time_metrics["discount_estimates"]:
        if estimate["front_end_delay_days"] != 0:
            continue
        annual = _annual_factor(estimate)
        if annual is not None:
            by_amount.setdefault(estimate["larger_amount"], []).append(annual)
    if len(by_amount) < 2:
        return 0
    amounts = sorted(by_amount)
    low = statistics.fmean(by_amount[amounts[0]])
    high = statistics.fmean(by_amount[amounts[-1]])
    difference = high - low
    if difference > 0.1:
        return 5
    if difference > 0.05:
        return 2.5
    return 0


def _risk_error(
    independence_metrics: dict[str, Any], outcomes: list[float]
) -> float | None:
    if len(outcomes) != 3:
        raise ValueError("Independence outcomes must contain low, middle, and high values")
    low, middle, high = outcomes
    deviations = []
    for point in independence_metrics["indifference_points"]:
        observed = point["indifference_probability"]
        if observed is None:
            continue
        expected_value = (
            point["reference_p_low"] * low
            + point["reference_p_middle"] * middle
            + point["reference_p_high"] * high
        )
        if point["axis"] == "y":
            denominator = high - middle
            predicted = (expected_value - middle) / denominator
        else:
            denominator = middle - low
            predicted = (middle - expected_value) / denominator
        predicted = min(1, max(0, predicted))
        deviations.append(abs(observed - predicted))
    return float(statistics.fmean(deviations) * 100) if deviations else None


def build_rationality_projection(
    independence: dict[str, Any], time: dict[str, Any]
) -> dict[str, Any]:
    """Build the established rationality dashboard object from canonical results."""
    independence_metadata = independence["metadata"]
    time_metadata = time["metadata"]
    if independence_metadata["model"]["id"] != time_metadata["model"]["id"]:
        raise ValueError("Independence and Time results belong to different models")
    if (
        independence_metadata["benchmark_version"] != time_metadata["benchmark_version"]
        or independence_metadata["schema_version"] != time_metadata["schema_version"]
    ):
        raise ValueError("Independence and Time result versions differ")

    delta = _patience(time["aggregate_metrics"])
    error = _risk_error(
        independence["aggregate_metrics"],
        independence_metadata["experiment"]["parameters"]["outcomes"],
    )
    penalty = _magnitude_penalty(time["aggregate_metrics"])
    return {
        "benchmark_version": independence_metadata["benchmark_version"],
        "schema_version": independence_metadata["schema_version"],
        "model": independence_metadata["model"]["id"],
        "metrics": {
            "patience": {
                "discount_factor": delta,
                "formatted_delta": f"{delta:.2f}" if delta is not None else "N/A",
            },
            "risk": {
                "error_rate": error,
                "formatted_error": f"{error:.0f}%" if error is not None else "N/A",
            },
            "penalties": {"magnitude_effect": penalty},
        },
    }
