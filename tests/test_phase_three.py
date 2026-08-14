"""Phase three acceptance tests for the canonical task engine."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from src.results.io import read_json, read_jsonl
from src.results.validation import validate_result_pair
from src.tasks import engine
from src.tasks.config import (
    active_experiments,
    canonical_run_paths,
    experiment_config,
)
from src.tasks.specs import fixed_trial_plans
from src.tasks.specs import next_trial_plan


def reduced_config(experiment_id):
    config = experiment_config(experiment_id)
    settings = config["settings"]
    for name in (
        "pool_amounts", "endowments", "coordination_payoffs", "safe_payoff_multipliers",
        "prizes", "final_payoff_levels", "queried_turns", "multipliers",
        "upper_bounds", "win_payoffs", "amounts", "delay_months",
        "front_end_delay_months", "receiver_sent_proportions",
    ):
        if name in settings:
            settings[name] = settings[name][:1]
    for name in (
        "repetitions_per_condition", "proposer_repetitions_per_condition",
        "responder_repetitions_per_condition",
    ):
        if name in settings:
            settings[name] = 1
    if experiment_id == "ultimatum":
        settings["offer_percentages"] = {"start": 0, "stop": 0, "step": 5}
    if experiment_id == "independence":
        settings["grid_divisions"] = 2
        settings["bisection_iterations"] = 2
    if experiment_id == "time":
        settings["bisection_iterations"] = 2
    return config


def test_canonical_paths_and_run_identifiers_are_shared(tmp_path):
    paths = canonical_run_paths(
        "gpt-4o", "dictator", "shared-run", release_root=tmp_path
    )
    assert paths.raw == tmp_path / "raw" / "gpt-4o" / "dictator" / "shared-run.jsonl"
    assert paths.derived == tmp_path / "derived" / "gpt-4o" / "dictator.json"


def test_monetary_scaling_at_all_protocol_levels():
    from src.tasks.centipede_game import generate_turns
    from src.tasks.travellers_dilemma import (
        monetary_bounds_for_level,
        monetary_increment_for_level,
    )

    for level in (10, 100, 1000):
        turns, final_payoffs = generate_turns(level)
        assert final_payoffs == pytest.approx((level, level / 2))
        assert next(turn for turn in turns if turn.turn_number == 5).take_payoff_you == (
            pytest.approx(level / 2)
        )
        low, high, bonus = monetary_bounds_for_level(level, 2, 100, 2)
        increment = monetary_increment_for_level(level, 100, 1)
        assert high == level
        assert low < high
        assert low / high == pytest.approx(0.02)
        assert bonus / high == pytest.approx(0.02)
        assert increment / high == pytest.approx(0.01)

    dictator = fixed_trial_plans(experiment_config("dictator"))
    for pool in (10, 100, 1000):
        plan = next(item for item in dictator if item.condition["pool_amount"] == pool)
        parsed = plan.parser(f"TRANSFER={pool / 4:g}")
        assert parsed.metrics["transfer_share"] == pytest.approx(0.25)


def test_strategic_game_prompts_define_actions_without_fixed_anchors():
    dictator = fixed_trial_plans(experiment_config("dictator"))[0].prompt
    assert "TRANSFER=<amount>" in dictator
    assert "$25" not in dictator and "$50" not in dictator

    ultimatum = fixed_trial_plans(experiment_config("ultimatum"))
    proposer = next(plan.prompt for plan in ultimatum if plan.role == "proposer")
    responder = next(plan.prompt for plan in ultimatum if plan.role == "responder")
    assert "OFFER=<amount>" in proposer
    assert "consider unfair" not in proposer
    assert "$25" not in proposer and "$50" not in proposer
    assert "DECISION=ACCEPT" in responder


def test_stag_hunt_prompt_gives_the_complete_symmetric_payoff_matrix():
    prompt = fixed_trial_plans(experiment_config("stag_hunt"))[0].prompt
    assert "Both choose A" in prompt
    assert "You choose A and the other person chooses B" in prompt
    assert "You choose B and the other person chooses A" in prompt
    assert "Both choose B" in prompt
    assert "CHOICE=A" in prompt and "CHOICE=B" in prompt


def test_trust_receiver_observes_endowment_and_sender_retained_amount():
    plans = fixed_trial_plans(experiment_config("trust_game"))
    zero_transfer_prompts = {
        plan.condition["endowment"]: plan.prompt
        for plan in plans
        if plan.role == "receiver" and plan.condition["sent_share"] == 0
    }
    assert len(set(zero_transfer_prompts.values())) == 3
    for endowment, prompt in zero_transfer_prompts.items():
        assert f"started with ${endowment:.2f}" in prompt
        assert f"kept ${endowment:.2f}" in prompt
        assert "RETURN=<amount>" in prompt


def test_beauty_and_public_goods_prompts_define_complete_response_rules():
    beauty = fixed_trial_plans(experiment_config("beauty_contest"))[0].prompt
    assert "split the prize equally" in beauty
    assert "CHOICE=<whole number>" in beauty

    public_goods = fixed_trial_plans(experiment_config("public_goods"))[0].prompt
    assert "at most two digits after the decimal point" in public_goods
    assert "CONTRIBUTION=<amount>" in public_goods


def test_invalid_response_is_visible_and_never_imputed(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "experiment_config", reduced_config)

    class InvalidModel:
        def generate_response(self, **_kwargs):
            return "undecided", None

    result = engine.run_experiment(
        "gpt-5.2", "dictator", run_id="invalid-run", interface=InvalidModel(),
        release_root=tmp_path, sleeper=lambda _seconds: None,
    )
    trial = result["raw"][0]["trial"]
    assert trial["validity"]["status"] == "invalid_response"
    assert trial["trial_metrics"] == {}
    assert result["derived"]["aggregate_metrics"]["overall_mean_transfer_share"] is None


def test_provider_failure_retries_and_cannot_become_a_choice(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "experiment_config", reduced_config)

    class FailingModel:
        calls = 0

        def generate_response(self, **_kwargs):
            self.calls += 1
            raise TimeoutError("offline provider failure")

    model = FailingModel()
    result = engine.run_experiment(
        "gpt-5.2", "dictator", run_id="failed-run", interface=model,
        release_root=tmp_path, sleeper=lambda _seconds: None,
    )
    trial = result["raw"][0]["trial"]
    assert model.calls == 3
    assert trial["validity"]["status"] == "provider_error"
    assert trial["transport"]["attempts"] == 3
    assert trial["response"]["raw_text"] is None
    assert trial["trial_metrics"] == {}


def test_interrupted_run_resumes_without_duplicate_valid_trials(monkeypatch, tmp_path):
    def resume_config(experiment_id):
        config = reduced_config(experiment_id)
        config["settings"]["repetitions_per_condition"] = 2
        return config

    monkeypatch.setattr(engine, "experiment_config", resume_config)

    class InterruptAfterOne:
        calls = 0

        def generate_response(self, **_kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise KeyboardInterrupt
                return "TRANSFER=1", None

    with pytest.raises(KeyboardInterrupt):
        engine.run_experiment(
            "gpt-5.2", "dictator", run_id="resume-run",
            interface=InterruptAfterOne(), release_root=tmp_path,
            sleeper=lambda _seconds: None,
        )

    paths = canonical_run_paths(
        "gpt-5.2", "dictator", "resume-run", release_root=tmp_path
    )
    interrupted = read_jsonl(paths.raw)
    first_timestamp = interrupted[0]["trial"]["completed_at"]
    assert [record["trial"]["validity"]["status"] for record in interrupted] == [
        "valid", "interrupted"
    ]

    result = engine.run_experiment(
        "gpt-5.2", "dictator", run_id="resume-run",
        interface=engine.FixtureModel(), resume=True, release_root=tmp_path,
        sleeper=lambda _seconds: None,
    )
    trials = [record["trial"] for record in result["raw"]]
    assert len(trials) == 2
    assert len({trial["trial_id"] for trial in trials}) == 2
    assert trials[0]["completed_at"] == first_timestamp
    assert result["derived"]["metadata"]["run"]["attempt"] == 2


def test_completed_resume_is_immutable(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "experiment_config", reduced_config)
    engine.run_experiment(
        "gpt-5.2", "dictator", run_id="completed-run",
        interface=engine.FixtureModel(), release_root=tmp_path,
        sleeper=lambda _seconds: None,
    )
    paths = canonical_run_paths(
        "gpt-5.2", "dictator", "completed-run", release_root=tmp_path
    )
    raw_before = paths.raw.read_bytes()
    derived_before = paths.derived.read_bytes()

    class MustNotRun:
        def generate_response(self, **_kwargs):
            raise AssertionError("completed resume requested a duplicate trial")

    engine.run_experiment(
        "gpt-5.2", "dictator", run_id="completed-run", interface=MustNotRun(),
        resume=True, release_root=tmp_path, sleeper=lambda _seconds: None,
    )
    assert paths.raw.read_bytes() == raw_before
    assert paths.derived.read_bytes() == derived_before


def test_repeated_trial_uncertainty_is_reproducible(monkeypatch, tmp_path):
    def uncertainty_config(experiment_id):
        config = reduced_config(experiment_id)
        config["settings"]["repetitions_per_condition"] = 2
        return config

    class AlternatingModel:
        calls = 0

        def generate_response(self, **_kwargs):
            self.calls += 1
            return ("TRANSFER=0" if self.calls == 1 else "TRANSFER=10"), None

    monkeypatch.setattr(engine, "experiment_config", uncertainty_config)
    result = engine.run_experiment(
        "gpt-5.2", "dictator", run_id="uncertainty-run",
        interface=AlternatingModel(), release_root=tmp_path,
        sleeper=lambda _seconds: None,
    )
    sample = result["derived"]["aggregate_metrics"]["sample"]
    assert sample["valid_rate_standard_error"] == 0
    assert sample["invalid_response_rate_standard_error"] == 0
    assert sample["primary_estimate_standard_error"] == pytest.approx(0.5)


def test_complete_simulated_batch_runs_all_active_experiments(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "experiment_config", reduced_config)
    results = engine.run_batch(
        "gpt-5.2", run_id="complete-offline-run", fixture=True,
        release_root=tmp_path, sleeper=lambda _seconds: None,
    )
    experiment_ids = [item["id"] for item in active_experiments()]
    assert [result["derived"]["metadata"]["experiment"]["id"] for result in results] == (
        experiment_ids
    )
    for experiment_id in experiment_ids:
        paths = canonical_run_paths(
            "gpt-5.2", experiment_id, "complete-offline-run", release_root=tmp_path
        )
        raw = read_jsonl(paths.raw)
        derived = read_json(paths.derived)
        assert validate_result_pair(raw, derived) == []
        metrics = derived["aggregate_metrics"]
        if experiment_id == "independence":
            assert metrics["validation"]["retests"] > 0
            assert metrics["diagnostics"]["monotonicity"]["checks"] > 0
            assert metrics["diagnostics"]["transitivity"]["checks"] == 0
            assert metrics["diagnostics"]["bidirectional"]["samples"] > 0
        if experiment_id == "time":
            assert metrics["validation"]["retests"] > 0
            assert metrics["diagnostics"]["monotonicity"]["checks"] > 0
            assert metrics["diagnostics"]["bidirectional"]["samples"] > 0


def test_canonical_planner_matches_the_frozen_release_call_count():
    fixture = engine.FixtureModel()
    total = 0
    by_experiment = {}
    for experiment in active_experiments():
        config = experiment_config(experiment["id"])
        fixed = [] if experiment["id"] in engine.BISECTION_EXPERIMENTS else fixed_trial_plans(config)
        records = []
        order_seed = f"20260811:gpt-5.2:{experiment['id']}"
        while plan := next_trial_plan(config, records, fixed, order_seed):
            response, _ = fixture.generate_response(prompt=plan.prompt)
            parsed = plan.parser(response)
            assert parsed is not None
            records.append({
                "trial": {
                    "trial_id": plan.trial_id,
                    "condition_id": plan.condition_id,
                    "condition": plan.condition,
                    "validity": {"status": "valid"},
                    "trial_metrics": parsed.metrics,
                    "prompt": {"text": plan.prompt},
                }
            })
            assert len(records) <= 8306
        by_experiment[experiment["id"]] = len(records)
        if experiment["id"] == "independence":
            primary_prompts = {
                record["trial"]["prompt"]["text"]
                for record in records
                if record["trial"]["condition"].get("phase") is None
            }
            transitivity_prompts = {
                record["trial"]["prompt"]["text"]
                for record in records
                if record["trial"]["condition"].get("phase")
                == "diagnostic_transitivity"
            }
            assert primary_prompts.isdisjoint(transitivity_prompts)
            transitivity_axes = [
                record["trial"]["condition"]["axis"]
                for record in records
                if record["trial"]["condition"].get("phase")
                == "diagnostic_transitivity"
            ]
            assert transitivity_axes.count("x") == 5
            assert transitivity_axes.count("y") == 5
        total += len(records)

    assert by_experiment["independence"] == 2787
    assert by_experiment["time"] == 3029
    assert total == 8306


def test_placeholders_are_removed_and_scripts_are_implemented():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "src" / "tasks" / "risk.py").exists()
    assert not (root / "src" / "tasks" / "transitivity.py").exists()
    for path in (root / "scripts" / "run_benchmark.py", root / "scripts" / "update_leaderboard.py"):
        source = path.read_text(encoding="utf-8")
        assert "Coming soon" not in source
        assert "def main" in source


def test_task_entry_points_share_the_state_free_engine():
    strategic = {
        "dictator", "ultimatum", "trust_game", "stag_hunt", "beauty_contest",
        "centipede_game", "public_goods", "travellers_dilemma", "matching_pennies",
    }
    for experiment in active_experiments():
        module = importlib.import_module(f"src.tasks.{experiment['id']}")
        main_source = inspect.getsource(module.main)
        assert "run_single_experiment_cli" in main_source
        if experiment["id"] in strategic:
            module_source = Path(module.__file__).read_text(encoding="utf-8")
            assert "\nllm =" not in module_source
            assert "\nPRINT_INTERACTIONS =" not in module_source


def test_leaderboard_updater_runs_both_canonical_projection_steps(monkeypatch, tmp_path):
    from scripts import update_leaderboard

    commands = []
    monkeypatch.setattr(
        update_leaderboard.subprocess,
        "run",
        lambda command, check: commands.append((command, check)),
    )
    assert update_leaderboard.main([
        "--release-root", str(tmp_path / "release"),
        "--output-dir", str(tmp_path / "web"),
        "--models", "gpt-4o",
    ]) == 0
    assert len(commands) == 2
    assert all(check is True for _command, check in commands)
    assert "generate_dashboard_data.py" in commands[0][0][1]
    assert "calculate_rationality_stats.py" in commands[1][0][1]
