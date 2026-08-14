"""Legacy-compatible dashboard projections from canonical results."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import read_json, read_jsonl, write_json
from .model_ids import model_id_to_path_component
from .validation import validate_result_pair


def dashboard_filename(experiment_id: str, model_id: str) -> str:
    """Return the established dashboard filename for one experiment and model."""
    model_key = model_id_to_path_component(model_id)
    if experiment_id == "independence":
        return f"independence_results_{model_key}.json"
    if experiment_id == "time":
        return f"time_experiment_{model_key}.json"
    return f"{experiment_id}_experiment_{model_key}.json"


def _valid_trials(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record["trial"]
        for record in raw_records
        if record["trial"]["validity"]["status"] == "valid"
    ]


def _base(derived: dict[str, Any]) -> dict[str, Any]:
    metadata = derived["metadata"]
    return {
        "benchmark_version": metadata["benchmark_version"],
        "schema_version": metadata["schema_version"],
        "model_id": metadata["model"]["id"],
        "timestamp": metadata["run"]["completed_at"],
    }


def _response(trial: dict[str, Any]) -> str:
    return trial["response"]["raw_text"] or ""


def _dictator_projection(
    raw_records: list[dict[str, Any]], derived: dict[str, Any]
) -> dict[str, Any]:
    base = _base(derived)
    metrics = derived["aggregate_metrics"]
    mean_share = metrics["overall_mean_transfer_share"]
    trials = []
    for trial in _valid_trials(raw_records):
        trials.append(
            {
                "pool_amount": trial["condition"]["pool_amount"],
                "offer_amount": trial["trial_metrics"]["transfer_amount"],
                "offer_percentage": trial["trial_metrics"]["transfer_share"] * 100,
                "raw_response": _response(trial),
                "trial_number": trial["repetition"],
                "timestamp": trial["completed_at"],
            }
        )
    percentage = mean_share * 100 if mean_share is not None else None
    return {
        **base,
        "tldr": f"Dictator Give: {percentage:.1f}%." if percentage is not None else "Dictator Give: N/A.",
        "analysis_text": "Generated from canonical Dictator trials.",
        "dictator_proposer": trials,
    }


def _ultimatum_projection(
    raw_records: list[dict[str, Any]], derived: dict[str, Any]
) -> dict[str, Any]:
    base = _base(derived)
    metrics = derived["aggregate_metrics"]
    proposer = []
    responder = []
    for trial in _valid_trials(raw_records):
        condition = trial["condition"]
        if trial["role"] == "proposer":
            proposer.append(
                {
                    "pool_amount": condition["pool_amount"],
                    "offer_amount": trial["trial_metrics"]["offer_amount"],
                    "offer_percentage": trial["trial_metrics"]["offer_share"] * 100,
                    "raw_response": _response(trial),
                    "trial_number": trial["repetition"],
                    "timestamp": trial["completed_at"],
                }
            )
        else:
            responder.append(
                {
                    "pool_amount": condition["pool_amount"],
                    "offer_amount": condition["offer_amount"],
                    "offer_percentage": condition["offer_share"] * 100,
                    "decision": "ACCEPT"
                    if trial["trial_metrics"]["accepted"]
                    else "REJECT",
                    "raw_response": _response(trial),
                    "trial_number": trial["repetition"],
                    "timestamp": trial["completed_at"],
                }
            )
    mean_share = metrics["overall_mean_offer_share"]
    percentage = mean_share * 100 if mean_share is not None else None
    return {
        **base,
        "tldr": f"Ultimatum Offer: {percentage:.1f}%." if percentage is not None else "Ultimatum Offer: N/A.",
        "analysis_text": "Generated from canonical Ultimatum trials.",
        "ultimatum_proposer": proposer,
        "ultimatum_responder": responder,
    }


def _trust_projection(
    raw_records: list[dict[str, Any]], derived: dict[str, Any]
) -> dict[str, Any]:
    base = _base(derived)
    metrics = derived["aggregate_metrics"]
    sender = []
    receiver = []
    for trial in _valid_trials(raw_records):
        condition = trial["condition"]
        values = trial["trial_metrics"]
        if trial["role"] == "sender":
            sender.append(
                {
                    "endowment": condition["endowment"],
                    "multiplier": condition["multiplier"],
                    "amount_sent": values["amount_sent"],
                    "send_rate": values["send_share"],
                    "raw_response": _response(trial),
                    "trial_number": trial["repetition"],
                    "timestamp": trial["completed_at"],
                }
            )
        else:
            sent = condition["sent_amount"]
            received = condition["received_amount"]
            receiver.append(
                {
                    "endowment": condition["endowment"],
                    "sent_amount": sent,
                    "multiplier": condition["multiplier"],
                    "received_amount": received,
                    "amount_returned": values["amount_returned"],
                    "return_rate_of_received": values["return_share_of_received"],
                    "return_rate_of_sent": values["return_multiple_of_sent"],
                    "raw_response": _response(trial),
                    "trial_number": trial["repetition"],
                    "timestamp": trial["completed_at"],
                }
            )
    send = metrics["overall_mean_send_share"]
    returned = metrics["overall_mean_return_share_of_received"]
    return {
        **base,
        "tldr_text": (
            f"Send Rate: {send * 100:.1f}%. Return Rate: {returned * 100:.1f}%."
            if send is not None and returned is not None
            else "Trust Game: N/A."
        ),
        "analysis_text": "Generated from canonical Trust Game trials.",
        "metrics": metrics,
        "sender_trials": sender,
        "receiver_trials": receiver,
    }


def _simple_trial_projection(
    raw_records: list[dict[str, Any]], derived: dict[str, Any]
) -> dict[str, Any]:
    experiment_id = derived["metadata"]["experiment"]["id"]
    projected = []
    for trial in _valid_trials(raw_records):
        condition = trial["condition"]
        metrics = trial["trial_metrics"]
        common = {
            "raw_response": _response(trial),
            "trial_number": trial["repetition"],
            "timestamp": trial["completed_at"],
        }
        if experiment_id == "stag_hunt":
            row = {
                "payoff": condition["coordination_payoff"],
                "x_multiplier": condition["safe_payoff_multiplier"],
                "decision": "B" if metrics["action"] == "stag" else "A",
            }
        elif experiment_id == "beauty_contest":
            row = {"prize": condition["prize"], "decision": metrics["guess"]}
        elif experiment_id == "centipede_game":
            row = {
                "magnitude": condition.get("magnitude", condition["final_payoff_level"] / 100),
                "monetary_level": condition["final_payoff_level"],
                "current_turn": condition["turn"],
                "current_turn_label": f"Turn {condition['turn']}",
                "take_payoff_you": condition.get("take_payoff_you"),
                "take_payoff_them": condition.get("take_payoff_them"),
                "final_payoff_you": condition.get("final_payoff_you"),
                "final_payoff_them": condition.get("final_payoff_them"),
                "decision": metrics["action"].upper(),
            }
        elif experiment_id == "public_goods":
            row = {
                "endowment": condition["endowment"],
                "multiplier": condition["multiplier"],
                "decision": metrics["contribution_amount"],
                "contribution_pct": metrics["contribution_share"],
            }
        elif experiment_id == "travellers_dilemma":
            row = {
                "magnitude": condition.get("magnitude", condition["upper_bound_level"] / 100),
                "monetary_level": condition["upper_bound_level"],
                "low": condition["lower_bound"],
                "high": condition["upper_bound"],
                "bonus": condition["bonus"],
                "decision": metrics["claim_amount"],
                "relative_claim": metrics["normalized_claim"],
                "claim_100_scale": metrics["claim_on_2_100_scale"],
            }
        elif experiment_id == "matching_pennies":
            row = {
                "win_payoff": condition["win_payoff"],
                "lose_payoff": condition["lose_payoff"],
                "decision": metrics["choice"].upper(),
            }
        else:
            raise ValueError(f"unsupported simple projection {experiment_id!r}")
        projected.append({**row, **common})

    metrics = derived["aggregate_metrics"]
    return {
        **_base(derived),
        "tldr_text": f"Canonical {experiment_id} results.",
        "analysis_text": f"Generated from canonical {experiment_id} trials.",
        "metrics": metrics,
        "trials": projected,
    }


def _independence_projection(derived: dict[str, Any]) -> dict[str, Any]:
    metrics = derived["aggregate_metrics"]
    results = []
    for point in metrics["indifference_points"]:
        results.append(
            {
                "reference_point": {
                    "p_L": point["reference_p_low"],
                    "p_M": point["reference_p_middle"],
                    "p_H": point["reference_p_high"],
                },
                "indifference_value": point["indifference_probability"],
                "axis": point["axis"].upper() if point["axis"] is not None else None,
            }
        )
    return {
        **_base(derived),
        "tldr_text": "Canonical Independence results.",
        "analysis_text": "Generated from canonical Independence trials.",
        "metrics": metrics,
        "results": results,
    }


def _time_projection(derived: dict[str, Any]) -> dict[str, Any]:
    metrics = derived["aggregate_metrics"]
    baseline = [
        estimate
        for estimate in metrics["discount_estimates"]
        if estimate["front_end_delay_days"] == 0
    ]
    delays = sorted({estimate["delay_days"] for estimate in baseline})
    by_amount: dict[float, dict[float, float | None]] = defaultdict(dict)
    for estimate in baseline:
        by_amount[estimate["larger_amount"]][estimate["delay_days"]] = estimate[
            "discount_factor"
        ]
    datasets = [
        {
            "label": f"${amount:g}",
            "data": [by_amount[amount].get(delay) for delay in delays],
        }
        for amount in sorted(by_amount)
    ]
    return {
        **_base(derived),
        "tldr_text": "Canonical Time results.",
        "analysis_text": "Generated from canonical Time trials.",
        "metrics": metrics,
        "labels": delays,
        "datasets": datasets,
    }


def build_dashboard_projection(
    raw_records: list[dict[str, Any]], derived: dict[str, Any]
) -> dict[str, Any]:
    """Build one dashboard projection from a validated canonical result pair."""
    experiment_id = derived["metadata"]["experiment"]["id"]
    if experiment_id == "dictator":
        return _dictator_projection(raw_records, derived)
    if experiment_id == "ultimatum":
        return _ultimatum_projection(raw_records, derived)
    if experiment_id == "trust_game":
        return _trust_projection(raw_records, derived)
    if experiment_id == "independence":
        return _independence_projection(derived)
    if experiment_id == "time":
        return _time_projection(derived)
    return _simple_trial_projection(raw_records, derived)


def generate_dashboard_file(
    raw_path: str | Path, derived_path: str | Path, output_dir: str | Path
) -> Path:
    """Validate canonical inputs and write one dashboard projection."""
    raw_records = read_jsonl(raw_path)
    derived = read_json(derived_path)
    findings = validate_result_pair(raw_records, derived)
    if findings:
        details = "; ".join(f"{item.code} {item.message}" for item in findings)
        raise ValueError(details)
    projection = build_dashboard_projection(raw_records, derived)
    metadata = derived["metadata"]
    path = Path(output_dir) / dashboard_filename(
        metadata["experiment"]["id"], metadata["model"]["id"]
    )
    write_json(path, projection)
    return path
