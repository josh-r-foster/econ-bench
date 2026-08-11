"""Hand-computed metric fixtures for the active experiments."""

import importlib

import pytest


def task(name):
    return importlib.import_module(f"src.tasks.{name}")


def test_independence_lottery_metrics():
    module = task("independence")
    point = module.TrianglePoint(p_L=0.2, p_H=0.3)
    assert point.p_M == pytest.approx(0.5)
    assert point.expected_value == pytest.approx(550.0)
    assert module.classify_lottery(point) == "better"
    assert module.classify_lottery(module.TrianglePoint(0.5, 0.1)) == "worse"
    assert len(module.generate_triangle_grid(2)) == 4


def test_time_preference_derived_metrics():
    module = task("time")
    result = module.DiscountRateResult(
        larger_amount=100,
        delay_days=365,
        front_end_delay=0,
        indifference_amount=80,
        implied_discount_factor=0.8,
        implied_annual_rate=0.25,
        n_iterations=10,
        choice_history=[],
        final_precision=0.1,
    )
    assert result.delay_ratio == pytest.approx(0.8)
    assert result.delay_months == pytest.approx(12.0)


def test_dictator_metrics():
    module = task("dictator")
    experiment = module.DictatorExperiment([100], 2)
    experiment.trials = [
        module.DictatorProposerTrial(100, 20, 20, "$20", 1),
        module.DictatorProposerTrial(100, 40, 40, "$40", 2),
    ]
    analysis = experiment.analyze()
    assert analysis["overall_mean_pct"] == pytest.approx(30)
    assert analysis["by_pool"][100] == pytest.approx(30)


def test_ultimatum_metrics():
    module = task("ultimatum")
    experiment = module.UltimatumExperiment([100], [10, 20], 2)
    experiment.proposer_trials = [
        module.UltimatumProposerTrial(100, 40, 40, "$40", 1),
        module.UltimatumProposerTrial(100, 60, 60, "$60", 2),
    ]
    experiment.responder_trials = [
        module.UltimatumResponderTrial(100, 10, 10, "REJECT", "REJECT", 1),
        module.UltimatumResponderTrial(100, 10, 10, "REJECT", "REJECT", 2),
        module.UltimatumResponderTrial(100, 20, 20, "ACCEPT", "ACCEPT", 1),
        module.UltimatumResponderTrial(100, 20, 20, "ACCEPT", "ACCEPT", 2),
    ]
    analysis = experiment.analyze()
    assert analysis["proposer_overall_mean_pct"] == pytest.approx(50)
    assert analysis["responder_mao_by_pool"][100] == 20


def test_trust_game_metrics():
    module = task("trust_game")
    experiment = module.TrustGameExperiment([100], 3, [0.5], 2)
    experiment.sender_trials = [
        module.TrustGameSenderTrial(100, 3, 25, 0.25, "$25", 1),
        module.TrustGameSenderTrial(100, 3, 75, 0.75, "$75", 2),
    ]
    experiment.receiver_trials = [
        module.TrustGameReceiverTrial(100, 50, 3, 150, 30, 0.2, 0.6, "$30", 1),
        module.TrustGameReceiverTrial(100, 50, 3, 150, 60, 0.4, 1.2, "$60", 2),
    ]
    analysis = experiment.analyze()
    assert analysis["sender_summary"]["average_send_rate"] == pytest.approx(50)
    assert analysis["receiver_summary"]["average_return_rate_of_received"] == pytest.approx(30)


def test_stag_hunt_metrics():
    module = task("stag_hunt")
    experiment = module.StagHuntExperiment([100], [0.5], 3)
    experiment.trials = [
        module.StagHuntTrial(100, 0.5, decision, decision, index)
        for index, decision in enumerate(["A", "B", "B"], start=1)
    ]
    analysis = experiment.analyze()
    assert analysis["summary"]["overall_cooperation_rate"] == pytest.approx(200 / 3)
    assert analysis["cooperation_by_payoff"][100] == pytest.approx(200 / 3)


def test_beauty_contest_metrics():
    module = task("beauty_contest")
    experiment = module.BeautyContestExperiment([100], 3)
    experiment.trials = [
        module.BeautyContestTrial(100, value, str(value), index)
        for index, value in enumerate([10, 20, 30], start=1)
    ]
    analysis = experiment.analyze()
    assert analysis["summary"]["overall_average_guess"] == pytest.approx(20)
    assert analysis["summary"]["overall_median_guess"] == pytest.approx(20)


def test_centipede_metrics():
    module = task("centipede_game")
    experiment = module.CentipedeGameExperiment([10], 2)
    experiment.trials = [
        module.CentipedeTrial(10, 10, 1, "Turn 1", 1.25, 0.62, 10, 5, "TAKE", "TAKE", 1),
        module.CentipedeTrial(10, 10, 1, "Turn 1", 1.25, 0.62, 10, 5, "PASS", "PASS", 2),
    ]
    analysis = experiment.analyze()
    assert analysis["summary"]["overall_take_rate"] == pytest.approx(50)
    assert analysis["by_magnitude"]["10"]["take_rate_by_turn"]["1"] == pytest.approx(50)


def test_public_goods_metrics():
    module = task("public_goods")
    experiment = module.PublicGoodsExperiment([100], [1.5], 10, 2)
    experiment.trials = [
        module.PublicGoodsTrial(100, 1.5, 0, 0, "$0", 1),
        module.PublicGoodsTrial(100, 1.5, 100, 1, "$100", 2),
    ]
    analysis = experiment.analyze()
    assert analysis["summary"]["overall_cooperation_rate"] == pytest.approx(50)
    cell = analysis["contribution_by_endowment_multiplier"]["100"]["1.5"]
    assert cell["average_contribution"] == pytest.approx(50)


def test_travellers_dilemma_metrics():
    module = task("travellers_dilemma")
    experiment = module.TravellersDilemmaExperiment([100], 2, 100, 2, 2)
    experiment.trials = [
        module.TravellersDilemmaTrial(100, 100, 2, 100, 2, 2, 0, 2, "2", 1),
        module.TravellersDilemmaTrial(100, 100, 2, 100, 2, 100, 1, 100, "100", 2),
    ]
    analysis = experiment.analyze()
    assert analysis["summary"]["overall_average_claim"] == pytest.approx(51)
    assert analysis["summary"]["lower_bound_rate"] == pytest.approx(50)


def test_matching_pennies_metrics():
    module = task("matching_pennies")
    experiment = module.MatchingPenniesExperiment([100], 0, 2)
    experiment.trials = [
        module.MatchingPenniesTrial(100, 0, "HEADS", "HEADS", 1),
        module.MatchingPenniesTrial(100, 0, "TAILS", "TAILS", 2),
    ]
    analysis = experiment.analyze()
    assert analysis["summary"]["heads_rate"] == pytest.approx(50)
    assert analysis["summary"]["distance_from_mixed_equilibrium"] == pytest.approx(0)
