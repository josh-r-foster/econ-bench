"""Protocol plans and metric adapters for active experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from src.tasks.response_formats import parse_labeled_choice


@dataclass(frozen=True)
class ParsedTrial:
    value: Any
    metrics: dict[str, Any]


@dataclass(frozen=True)
class TrialPlan:
    trial_id: str
    condition_id: str
    condition: dict[str, Any]
    repetition: int
    role: str | None
    prompt: str
    parser_name: str
    parser: Callable[[str], ParsedTrial | None]


def _key(value: float | int) -> str:
    return f"{value:g}".replace(".", "p")


def _ab(response: str) -> str | None:
    return parse_labeled_choice(
        response, choices=("A", "B"), labels=("choice",)
    )


def fixed_trial_plans(
    config: dict[str, Any], seed: int | str | None = None
) -> list[TrialPlan]:
    experiment_id = config["id"]
    settings = config["settings"]
    repetitions = settings.get("repetitions_per_condition", 10)
    plans: list[TrialPlan] = []

    if experiment_id == "dictator":
        from src.tasks import dictator as module

        for pool in settings["pool_amounts"]:
            condition_id = f"pool-{_key(pool)}"
            for repetition in range(1, repetitions + 1):
                def parse(response, pool=pool):
                    amount = module.parse_dollar_amount(response, pool)
                    if amount is None or not 0 <= amount <= pool:
                        return None
                    return ParsedTrial(amount, {
                        "transfer_amount": amount,
                        "transfer_share": amount / pool,
                    })

                plans.append(TrialPlan(
                    f"{condition_id}-r{repetition:03d}", condition_id,
                    {"pool_amount": pool}, repetition, None,
                    module.dictator_proposer_prompt(pool), "parse_dollar_amount", parse,
                ))

    elif experiment_id == "ultimatum":
        from src.tasks import ultimatum as module

        proposer_repetitions = settings["proposer_repetitions_per_condition"]
        responder_repetitions = settings["responder_repetitions_per_condition"]
        offers = settings["offer_percentages"]
        offer_percentages = range(offers["start"], offers["stop"] + 1, offers["step"])
        for pool in settings["pool_amounts"]:
            condition_id = f"proposer-pool-{_key(pool)}"
            for repetition in range(1, proposer_repetitions + 1):
                def parse(response, pool=pool):
                    amount = module.parse_dollar_amount(response, pool)
                    if amount is None or not 0 <= amount <= pool:
                        return None
                    return ParsedTrial(amount, {
                        "role": "proposer",
                        "offer_amount": amount,
                        "offer_share": amount / pool,
                    })

                plans.append(TrialPlan(
                    f"{condition_id}-r{repetition:03d}", condition_id,
                    {"pool_amount": pool}, repetition, "proposer",
                    module.ultimatum_proposer_prompt(pool), "parse_dollar_amount", parse,
                ))
            for percentage in offer_percentages:
                offer_amount = pool * percentage / 100
                offer_share = percentage / 100
                condition_id = f"responder-pool-{_key(pool)}-offer-{_key(offer_share)}"
                for repetition in range(1, responder_repetitions + 1):
                    def parse(response):
                        decision = module.parse_accept_reject(response)
                        if decision is None:
                            return None
                        return ParsedTrial(decision, {
                            "role": "responder",
                            "accepted": decision == "ACCEPT",
                        })

                    plans.append(TrialPlan(
                        f"{condition_id}-r{repetition:03d}", condition_id,
                        {"pool_amount": pool, "offer_amount": offer_amount,
                         "offer_share": offer_share}, repetition, "responder",
                        module.ultimatum_responder_prompt(pool, offer_amount),
                        "parse_accept_reject", parse,
                    ))

    elif experiment_id == "trust_game":
        from src.tasks import trust_game as module

        multiplier = settings["multiplier"]
        for endowment in settings["endowments"]:
            condition_id = f"sender-endowment-{_key(endowment)}"
            for repetition in range(1, repetitions + 1):
                def parse(response, endowment=endowment):
                    amount = module.parse_dollar_amount(
                        response, endowment, "send"
                    )
                    if amount is None:
                        return None
                    return ParsedTrial(amount, {
                        "role": "sender", "amount_sent": amount,
                        "send_share": amount / endowment,
                    })

                plans.append(TrialPlan(
                    f"{condition_id}-r{repetition:03d}", condition_id,
                    {"endowment": endowment, "multiplier": multiplier},
                    repetition, "sender",
                    module.TrustGamePrompts.sender_prompt(endowment, multiplier),
                    "parse_dollar_amount", parse,
                ))
            for sent_share in settings["receiver_sent_proportions"]:
                sent = endowment * sent_share
                received = sent * multiplier
                condition_id = (
                    f"receiver-endowment-{_key(endowment)}-sent-{_key(sent_share)}"
                )
                for repetition in range(1, repetitions + 1):
                    def parse(response, sent=sent, received=received):
                        amount = module.parse_dollar_amount(
                            response, received, "return"
                        )
                        if amount is None:
                            return None
                        return ParsedTrial(amount, {
                            "role": "receiver", "amount_returned": amount,
                            "return_share_of_received": amount / received if received else None,
                            "return_multiple_of_sent": amount / sent if sent else None,
                        })

                    plans.append(TrialPlan(
                        f"{condition_id}-r{repetition:03d}", condition_id,
                        {"endowment": endowment, "multiplier": multiplier,
                         "sent_amount": sent, "sent_share": sent_share,
                         "received_amount": received}, repetition, "receiver",
                        module.TrustGamePrompts.receiver_prompt(
                            endowment, sent, multiplier
                        ),
                        "parse_dollar_amount", parse,
                    ))

    elif experiment_id == "stag_hunt":
        from src.tasks import stag_hunt as module

        for payoff in settings["coordination_payoffs"]:
            for multiplier in settings["safe_payoff_multipliers"]:
                condition_id = f"payoff-{_key(payoff)}-safe-{_key(multiplier)}"
                first_safe_label = random.Random(
                    f"{seed}:{condition_id}:labels"
                ).choice(("A", "B"))
                for repetition in range(1, repetitions + 1):
                    safe_label = (
                        first_safe_label
                        if repetition % 2 == 1
                        else ("B" if first_safe_label == "A" else "A")
                    )
                    dominant_label = "B" if safe_label == "A" else "A"

                    def parse(response, dominant_label=dominant_label):
                        choice = module.parse_a_b(response)
                        if choice is None:
                            return None
                        action = "stag" if choice == dominant_label else "hare"
                        return ParsedTrial(choice, {
                            "action": action,
                            "payoff_dominant_choice": action == "stag",
                        })

                    plans.append(TrialPlan(
                        f"{condition_id}-r{repetition:03d}", condition_id,
                        {"coordination_payoff": payoff,
                         "safe_payoff_multiplier": multiplier,
                         "safe_action_label": safe_label,
                         "payoff_dominant_action_label": dominant_label},
                        repetition, None,
                        module.StagHuntPrompts.generic_stag_hunt(
                            payoff, multiplier, settings["miscoordination_payoff"],
                            safe_label=safe_label,
                        ), "parse_a_b", parse,
                    ))

    elif experiment_id == "beauty_contest":
        from src.tasks import beauty_contest as module

        low = settings["choice_lower_bound"]
        high = settings["choice_upper_bound"]
        for prize in settings["prizes"]:
            condition_id = f"prize-{_key(prize)}"
            for repetition in range(1, repetitions + 1):
                def parse(response):
                    guess = module.parse_number(response, low, high)
                    return None if guess is None else ParsedTrial(guess, {
                        "guess": guess, "distance_from_nash": abs(guess),
                    })

                plans.append(TrialPlan(
                    f"{condition_id}-r{repetition:03d}", condition_id,
                    {"prize": prize, "choice_lower_bound": low,
                     "choice_upper_bound": high,
                     "target_fraction": settings["target_fraction"]},
                    repetition, None,
                    module.BeautyContestPrompts.generic_game(
                        prize, settings["other_players"], low, high,
                        settings["target_fraction"]
                    ), "parse_number", parse,
                ))

    elif experiment_id == "centipede_game":
        from src.tasks import centipede_game as module

        for level in settings["final_payoff_levels"]:
            turns, final_payoffs = module.generate_turns(level)
            tree = module.format_game_tree(turns, final_payoffs)
            for turn_number in settings["queried_turns"]:
                turn = next(item for item in turns if item.turn_number == turn_number)
                condition_id = f"level-{_key(level)}-turn-{turn_number}"
                for repetition in range(1, repetitions + 1):
                    def parse(response):
                        choice = module.parse_pass_take(response)
                        if choice is None:
                            return None
                        action = choice.lower()
                        return ParsedTrial(choice, {
                            "action": action,
                            "backward_induction_consistent": action == "take",
                        })

                    plans.append(TrialPlan(
                        f"{condition_id}-r{repetition:03d}", condition_id,
                        {"final_payoff_level": level, "turn": turn_number,
                         "take_payoff_you": turn.take_payoff_you,
                         "take_payoff_them": turn.take_payoff_them,
                         "final_payoff_you": final_payoffs[0],
                         "final_payoff_them": final_payoffs[1]}, repetition, None,
                        module.CentipedeGamePrompts.generic_game(
                            tree, module.format_turn_label(turn_number)
                        ), "parse_pass_take", parse,
                    ))

    elif experiment_id == "public_goods":
        from src.tasks import public_goods as module

        for endowment in settings["endowments"]:
            for multiplier in settings["multipliers"]:
                condition_id = f"endowment-{_key(endowment)}-multiplier-{_key(multiplier)}"
                for repetition in range(1, repetitions + 1):
                    def parse(response, endowment=endowment):
                        amount = module.parse_contribution(response, endowment)
                        return None if amount is None else ParsedTrial(amount, {
                            "contribution_amount": amount,
                            "contribution_share": amount / endowment,
                        })

                    plans.append(TrialPlan(
                        f"{condition_id}-r{repetition:03d}", condition_id,
                        {"endowment": endowment, "multiplier": multiplier},
                        repetition, None,
                        module.PublicGoodsPrompts.generic_game(
                            endowment, multiplier, settings["players"]
                        ), "parse_contribution", parse,
                    ))

    elif experiment_id == "travellers_dilemma":
        from src.tasks import travellers_dilemma as module

        for level in settings["upper_bounds"]:
            low, high, bonus = module.monetary_bounds_for_level(
                level, settings["base_lower_bound"], settings["base_upper_bound"],
                settings["base_bonus"]
            )
            increment = module.monetary_increment_for_level(
                level,
                settings["base_upper_bound"],
                settings["base_claim_increment"],
            )
            condition_id = f"level-{_key(level)}"
            for repetition in range(1, repetitions + 1):
                def parse(response, low=low, high=high, increment=increment):
                    claim = module.parse_number(response, low, high, increment)
                    if claim is None:
                        return None
                    normalized = (claim - low) / (high - low)
                    return ParsedTrial(claim, {
                        "claim_amount": claim, "normalized_claim": normalized,
                        "claim_on_2_100_scale": module.claim_on_100_scale(normalized),
                        "lower_bound_choice": claim == low,
                    })

                plans.append(TrialPlan(
                    f"{condition_id}-r{repetition:03d}", condition_id,
                    {"upper_bound_level": level, "lower_bound": low,
                     "upper_bound": high, "bonus": bonus,
                     "claim_increment": increment}, repetition, None,
                    module.TravellersDilemmaPrompts.generic_game(
                        low, high, bonus, increment
                    ),
                    "parse_number", parse,
                ))

    elif experiment_id == "matching_pennies":
        from src.tasks import matching_pennies as module

        lose = settings["lose_payoff"]
        roles = settings.get("roles", ["matching", "mismatching"])
        for win in settings["win_payoffs"]:
            for role in roles:
                condition_id = f"win-{_key(win)}-role-{role}"
                first_order = random.Random(
                    f"{seed}:{condition_id}:labels"
                ).choice(("heads_first", "tails_first"))
                for repetition in range(1, repetitions + 1):
                    order_name = (
                        first_order
                        if repetition % 2 == 1
                        else (
                            "tails_first"
                            if first_order == "heads_first"
                            else "heads_first"
                        )
                    )
                    choice_order = (
                        ("HEADS", "TAILS")
                        if order_name == "heads_first"
                        else ("TAILS", "HEADS")
                    )

                    def parse(response):
                        choice = module.parse_heads_tails(response)
                        return None if choice is None else ParsedTrial(
                            choice, {"choice": choice.lower()}
                        )

                    plans.append(TrialPlan(
                        f"{condition_id}-r{repetition:03d}", condition_id,
                        {"win_payoff": win, "lose_payoff": lose,
                         "payoff_role": role, "choice_order": order_name},
                        repetition, role,
                        module.MatchingPenniesPrompts.generic_game(
                            win, lose, role=role, choice_order=choice_order
                        ),
                        "parse_heads_tails", parse,
                    ))
    else:
        raise ValueError(f"fixed trial plans do not support {experiment_id!r}")
    if seed is not None:
        random.Random(f"{seed}:{experiment_id}:fixed-order").shuffle(plans)
    return plans


def bisection_conditions(
    config: dict[str, Any], seed: int | str | None = None
) -> list[dict[str, Any]]:
    settings = config["settings"]
    if config["id"] == "independence":
        divisions = settings["grid_divisions"]
        conditions = []
        for low_index in range(divisions + 1):
            for high_index in range(divisions + 1 - low_index):
                p_low = low_index / divisions
                p_high = high_index / divisions
                if (p_low, p_high) in {(0, 0), (1, 0), (0, 1)}:
                    continue
                p_middle = max(0.0, 1 - p_low - p_high)
                outcomes = settings["outcomes"]
                expected = p_low * outcomes[0] + p_middle * outcomes[1] + p_high * outcomes[2]
                axis = "y" if expected >= outcomes[1] else "x"
                conditions.append({
                    "condition_id": f"axis-{axis}-pl-{_key(p_low)}-ph-{_key(p_high)}",
                    "axis": axis, "reference_p_low": p_low,
                    "reference_p_middle": p_middle, "reference_p_high": p_high,
                })
        if seed is not None:
            random.Random(f"{seed}:independence:condition-order").shuffle(
                conditions
            )
            for index, condition in enumerate(conditions):
                condition["swap_order"] = bool(index % 2)
        return conditions
    if config["id"] == "time":
        days_per_month = settings["days_per_month"]
        conditions = [
            {
                "condition_id": (
                    f"amount-{_key(amount)}-delay-{_key(round(delay * days_per_month))}"
                    f"-front-{_key(round(front * days_per_month))}"
                ),
                "larger_amount": amount,
                "delay_days": round(delay * days_per_month),
                "front_end_delay_days": round(front * days_per_month),
            }
            for amount in settings["amounts"]
            for delay in settings["delay_months"]
            for front in settings["front_end_delay_months"]
        ]
        if seed is not None:
            random.Random(f"{seed}:time:condition-order").shuffle(conditions)
            for index, condition in enumerate(conditions):
                condition["swap_order"] = bool(index % 2)
        return conditions
    raise ValueError(f"bisection conditions do not support {config['id']!r}")


def _selected_conditions(
    config: dict[str, Any], *, seed: int | str, phase: str
) -> list[dict[str, Any]]:
    settings = config["settings"]
    conditions = bisection_conditions(config, seed)
    if phase == "validation":
        count = max(1, int(len(conditions) * settings["validation_fraction"]))
        eligible = conditions
    elif phase == "diagnostic_bidirectional":
        count = settings["diagnostic_bidirectional_sequences"]
        if config["id"] == "independence":
            eligible = [item for item in conditions if item["axis"] == "x"]
        else:
            eligible = [
                item for item in conditions if item["front_end_delay_days"] == 0
            ]
    else:
        raise ValueError(f"unknown supplemental bisection phase {phase!r}")

    count = min(count, len(eligible))
    generator = random.Random(f"{seed}:{config['id']}:{phase}")
    selected_ids = {
        item["condition_id"] for item in generator.sample(eligible, count)
    }
    selected = []
    for item in eligible:
        if item["condition_id"] not in selected_ids:
            continue
        source_id = item["condition_id"]
        supplemental = dict(item)
        supplemental["condition_id"] = f"{phase}-{source_id}"
        supplemental["phase"] = phase
        supplemental["source_condition_id"] = source_id
        supplemental["swap_order"] = (
            bool(item.get("swap_order", False))
            if phase == "validation"
            else not bool(item.get("swap_order", False))
        )
        selected.append(supplemental)
    return selected


def validation_bisection_conditions(
    config: dict[str, Any], seed: int | str
) -> list[dict[str, Any]]:
    return _selected_conditions(config, seed=seed, phase="validation")


def bidirectional_bisection_conditions(
    config: dict[str, Any], seed: int | str
) -> list[dict[str, Any]]:
    return _selected_conditions(
        config, seed=seed, phase="diagnostic_bidirectional"
    )


def _lottery_description(
    p_low: float, p_high: float, outcomes: list[float]
) -> str:
    p_middle = max(0.0, 1 - p_low - p_high)
    probabilities = (p_high, p_middle, p_low)
    displayed = [round(probability * 100, 1) for probability in probabilities]
    residual_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    displayed[residual_index] = round(
        displayed[residual_index] + 100.0 - sum(displayed), 1
    )
    return ", ".join(
        f"{probability:.1f}% chance of ${outcome:g}"
        for probability, outcome in zip(
            displayed, (outcomes[2], outcomes[1], outcomes[0])
        )
    )


def _lottery_prompt(
    option_a: tuple[float, float],
    option_b: tuple[float, float],
    outcomes: list[float],
) -> str:
    return (
        "You must choose between two lotteries. Which do you prefer?\n\n"
        f"Option A: {_lottery_description(*option_a, outcomes)}\n"
        f"Option B: {_lottery_description(*option_b, outcomes)}\n\n"
        "Return one line using CHOICE=A or CHOICE=B.\n\nYour choice"
    )


def _completed_bisection_value(
    trials: list[dict[str, Any]], experiment_id: str
) -> float | None:
    if not trials or any(
        trial["validity"]["status"] != "valid" for trial in trials
    ):
        return None
    lower = 0.0
    first = trials[0]["condition"]
    upper = 1.0 if experiment_id == "independence" else first["larger_amount"]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for trial in trials:
        grouped.setdefault(
            trial["condition"]["bisection_iteration"], []
        ).append(trial)
    expected_iterations = max(grouped)
    for iteration in range(1, expected_iterations + 1):
        step = grouped.get(iteration, [])
        if not step:
            return None
        repetitions = step[0]["condition"].get(
            "bisection_repetitions_per_step", 1
        )
        if len(step) != repetitions:
            return None
        choices = [trial["trial_metrics"]["semantic_choice"] for trial in step]
        choice = max(set(choices), key=choices.count)
        if choices.count(choice) <= repetitions // 2:
            return None
        midpoint = step[0]["condition"]["midpoint"]
        if experiment_id == "time":
            if choice == "sooner":
                upper = midpoint
            else:
                lower = midpoint
        elif step[0]["condition"]["axis"] == "y":
            if choice == "reference_lottery":
                lower = midpoint
            else:
                upper = midpoint
        elif choice == "reference_lottery":
            upper = midpoint
        else:
            lower = midpoint
    return (lower + upper) / 2


def diagnostic_trial_plans(
    config: dict[str, Any], records: list[dict[str, Any]]
) -> list[TrialPlan]:
    settings = config["settings"]
    plans: list[TrialPlan] = []

    if config["id"] == "independence":
        outcomes = settings["outcomes"]
        monotonicity_pairs = [
            ((0.1, 0.6), (0.3, 0.4)),
            ((0.0, 0.8), (0.2, 0.6)),
            ((0.2, 0.4), (0.4, 0.2)),
            ((0.0, 0.5), (0.0, 0.3)),
            ((0.3, 0.0), (0.5, 0.0)),
        ][:settings["diagnostic_monotonicity_checks"]]
        for index, (option_a, option_b) in enumerate(monotonicity_pairs, start=1):
            condition_id = f"diagnostic-monotonicity-{index:03d}"
            swapped = index % 2 == 0
            presented_a, presented_b = (
                (option_b, option_a) if swapped else (option_a, option_b)
            )

            def parse(response):
                choice = _ab(response)
                if choice is None:
                    return None
                semantic = "option_a" if choice == "A" else "option_b"
                return ParsedTrial(choice, {"semantic_choice": semantic})

            plans.append(TrialPlan(
                condition_id, condition_id,
                {
                    "phase": "diagnostic_monotonicity",
                    "expected_semantic_choice": "option_b" if swapped else "option_a",
                    "option_a_p_low": presented_a[0],
                    "option_a_p_high": presented_a[1],
                    "option_b_p_low": presented_b[0],
                    "option_b_p_high": presented_b[1],
                },
                1, None, _lottery_prompt(presented_a, presented_b, outcomes),
                "parse_ab_choice", parse,
            ))

        groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            trial = record["trial"]
            if trial["condition"].get("phase") is None:
                groups.setdefault(trial["condition_id"], []).append(trial)
        ranked: dict[str, list[tuple[float, str, dict[str, Any]]]] = {
            "x": [], "y": []
        }
        for source_id, group in groups.items():
            value = _completed_bisection_value(group, "independence")
            if value is not None:
                ranked[group[0]["condition"]["axis"]].append(
                    (value, source_id, group[0]["condition"])
                )
        limit = settings["diagnostic_transitivity_checks"]
        pairs_by_axis: dict[str, list[Any]] = {"x": [], "y": []}
        for axis in ("x", "y"):
            ordered = sorted(ranked[axis], key=lambda item: item[0])
            for distance in range(1, len(ordered)):
                for index in range(len(ordered) - distance):
                    pairs_by_axis[axis].append(
                        (ordered[index + distance], ordered[index])
                        if axis == "y"
                        else (ordered[index], ordered[index + distance])
                    )
        pairs = []
        for index in range(max(map(len, pairs_by_axis.values()), default=0)):
            for axis in ("x", "y"):
                if index < len(pairs_by_axis[axis]):
                    pairs.append((axis, *pairs_by_axis[axis][index]))
        existing_prompts = {
            prompt
            for record in records
            if (prompt := record["trial"].get("prompt", {}).get("text"))
        }
        diagnostic_index = 0
        for axis, (_, source_a, choice_a), (_, source_b, choice_b) in pairs:
            if diagnostic_index >= limit:
                break
            option_a = (choice_a["reference_p_low"], choice_a["reference_p_high"])
            option_b = (choice_b["reference_p_low"], choice_b["reference_p_high"])
            prompt = _lottery_prompt(option_a, option_b, outcomes)
            expected = "option_a"
            if prompt in existing_prompts:
                prompt = _lottery_prompt(option_b, option_a, outcomes)
                expected = "option_b"
            if prompt in existing_prompts:
                continue
            diagnostic_index += 1
            condition_id = f"diagnostic-transitivity-{diagnostic_index:03d}"
            existing_prompts.add(prompt)

            def parse(response):
                choice = _ab(response)
                if choice is None:
                    return None
                semantic = "option_a" if choice == "A" else "option_b"
                return ParsedTrial(choice, {"semantic_choice": semantic})

            plans.append(TrialPlan(
                condition_id, condition_id,
                {
                    "phase": "diagnostic_transitivity",
                    "expected_semantic_choice": expected,
                    "axis": axis,
                    "option_a_source_condition_id": source_a,
                    "option_b_source_condition_id": source_b,
                },
                1, None, prompt,
                "parse_ab_choice", parse,
            ))

    elif config["id"] == "time":
        from src.tasks.time import DiscountRatePrompts

        cases = [
            (50, 100, 7, "either"),
            (80, 100, 30, "either"),
            (95, 100, 7, "either"),
            (100, 100, 30, "sooner"),
            (110, 100, 30, "sooner"),
        ][:settings["diagnostic_monotonicity_checks"]]
        for index, (sooner, later, delay, expected) in enumerate(cases, start=1):
            condition_id = f"diagnostic-monotonicity-{index:03d}"
            swapped = index % 2 == 0

            def parse(response, swapped=swapped):
                choice = _ab(response)
                if choice is None:
                    return None
                chose_sooner = choice == ("B" if swapped else "A")
                semantic = "sooner" if chose_sooner else "later"
                return ParsedTrial(choice, {"semantic_choice": semantic})

            plans.append(TrialPlan(
                condition_id, condition_id,
                {
                    "phase": "diagnostic_monotonicity",
                    "sooner_amount": sooner,
                    "later_amount": later,
                    "delay_days": delay,
                    "expected_semantic_choice": expected,
                },
                1, None,
                DiscountRatePrompts.binary_choice(sooner, later, 0, delay, swapped),
                "parse_ab_choice", parse,
            ))
    return plans


def bisection_plan(
    config: dict[str, Any], base: dict[str, Any], existing_trials: list[dict[str, Any]]
) -> TrialPlan | None:
    settings = config["settings"]
    iterations = settings["bisection_iterations"]
    repetitions = settings.get("responses_per_bisection_step", 1)
    if repetitions < 1 or repetitions % 2 == 0:
        raise ValueError("bisection responses per step must be a positive odd number")
    if any(trial["validity"]["status"] != "valid" for trial in existing_trials):
        return None
    if len(existing_trials) >= iterations * repetitions:
        return None
    lower = 0.0
    upper = 1.0 if config["id"] == "independence" else base["larger_amount"]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for trial in existing_trials:
        grouped.setdefault(
            trial["condition"]["bisection_iteration"], []
        ).append(trial)
    iteration = 1
    while iteration <= iterations and len(grouped.get(iteration, [])) == repetitions:
        step = grouped[iteration]
        choices = [trial["trial_metrics"]["semantic_choice"] for trial in step]
        choice = max(set(choices), key=choices.count)
        midpoint = step[0]["condition"]["midpoint"]
        if config["id"] == "time":
            if choice == "sooner":
                upper = midpoint
            else:
                lower = midpoint
        elif base["axis"] == "y":
            if choice == "reference_lottery":
                lower = midpoint
            else:
                upper = midpoint
        elif choice == "reference_lottery":
            upper = midpoint
        else:
            lower = midpoint
        iteration += 1
    if iteration > iterations:
        return None
    current_step = grouped.get(iteration, [])
    if len(current_step) >= repetitions:
        raise ValueError("bisection step contains too many responses")
    repetition = len(current_step) + 1
    midpoint = (lower + upper) / 2
    condition = {
        key: value for key, value in base.items() if key != "condition_id"
    }
    condition.update({
        "bisection_iteration": iteration,
        "bisection_repetition": repetition,
        "bisection_repetitions_per_step": repetitions,
        "lower_bound_before": lower,
        "upper_bound_before": upper,
        "midpoint": midpoint,
    })
    condition_id = base["condition_id"]
    trial_id = f"{condition_id}-i{iteration:02d}-r{repetition:02d}"

    if config["id"] == "independence":
        from src.tasks.independence import MMTrianglePrompts, TrianglePoint

        point = TrianglePoint(base["reference_p_low"], base["reference_p_high"])
        swapped = bool(base.get("swap_order", False))
        prompt = MMTrianglePrompts.binary_choice(
            point, midpoint, base["axis"].upper(), swapped
        )

        def parse(response):
            choice = _ab(response)
            if choice is None:
                return None
            chose_reference = choice == ("B" if swapped else "A")
            semantic = "reference_lottery" if chose_reference else "axis_lottery"
            return ParsedTrial(choice, {"semantic_choice": semantic})
    else:
        from src.tasks.time import DiscountRatePrompts

        swapped = bool(base.get("swap_order", False))
        prompt = DiscountRatePrompts.binary_choice(
            midpoint, base["larger_amount"], base["front_end_delay_days"],
            base["front_end_delay_days"] + base["delay_days"], swapped
        )

        def parse(response):
            choice = _ab(response)
            if choice is None:
                return None
            chose_sooner = choice == ("B" if swapped else "A")
            return ParsedTrial(
                choice, {"semantic_choice": "sooner" if chose_sooner else "later"}
            )

    return TrialPlan(
        trial_id, condition_id, condition, repetition, None, prompt,
        "parse_ab_choice", parse
    )
