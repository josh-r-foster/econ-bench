"""Tests for manifest-backed task configuration and result locations."""

import importlib
import inspect
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_manifest():
    with (PROJECT_ROOT / "config" / "experiments.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def test_every_active_task_has_its_configured_parser_and_temperature():
    manifest = load_manifest()
    for experiment in manifest["experiments"]:
        module_name = experiment["script"].removesuffix(".py").replace("/", ".")
        module = importlib.import_module(module_name)
        parser_names = experiment.get("response_parsers", [experiment.get("response_parser")])
        for parser_name in parser_names:
            assert callable(getattr(module, parser_name, None)), (
                f"{experiment['id']} config names missing parser {parser_name!r}"
            )

        default_temperature = inspect.signature(module.generate_response).parameters[
            "temperature"
        ].default
        assert default_temperature == experiment["temperature"], (
            f"{experiment['id']} requests temperature {experiment['temperature']} "
            f"but its task defaults to {default_temperature}"
        )


def test_manifest_conditions_match_task_constants():
    experiments = {item["id"]: item["settings"] for item in load_manifest()["experiments"]}

    independence = importlib.import_module("src.tasks.independence")
    assert [independence.X_LOW, independence.X_MID, independence.X_HIGH] == experiments[
        "independence"
    ]["outcomes"]

    time = importlib.import_module("src.tasks.time")
    assert time.AMOUNTS == experiments["time"]["amounts"]
    assert [round(delay / 30.42) for delay in time.DELAYS] == experiments["time"][
        "delay_months"
    ]
    assert [round(delay / 30.42) for delay in time.FRONT_END_DELAYS] == experiments[
        "time"
    ]["front_end_delay_months"]
    assert time.N_ITERATIONS == experiments["time"]["bisection_iterations"]

    expectations = {
        "dictator": ("POOL_AMOUNTS", "pool_amounts"),
        "ultimatum": ("POOL_AMOUNTS", "pool_amounts"),
        "trust_game": ("ENDOWMENTS", "endowments"),
        "stag_hunt": ("PAYOFFS", "coordination_payoffs"),
        "beauty_contest": ("PRIZES", "prizes"),
        "centipede_game": ("MONETARY_LEVELS", "final_payoff_levels"),
        "public_goods": ("ENDOWMENTS", "endowments"),
        "travellers_dilemma": ("MONETARY_LEVELS", "upper_bounds"),
        "matching_pennies": ("WIN_PAYOFFS", "win_payoffs"),
    }
    for experiment_id, (constant_name, setting_name) in expectations.items():
        module = importlib.import_module(f"src.tasks.{experiment_id}")
        assert list(getattr(module, constant_name)) == experiments[experiment_id][setting_name]


def test_canonical_output_templates_resolve_inside_the_release_directory():
    shared = load_manifest()["shared_settings"]
    locations = shared["canonical_locations"]
    values = {
        "model_id": "gpt-4o",
        "experiment_id": "dictator",
        "run_id": "run-001",
    }

    assert locations["raw"].format(**values) == (
        "data/releases/1.0.0/raw/gpt-4o/dictator/run-001.jsonl"
    )
    assert locations["derived"].format(**values) == (
        "data/releases/1.0.0/derived/gpt-4o/dictator.json"
    )
    assert locations["release_manifest"] == "data/releases/1.0.0/manifest.json"
    assert locations["dashboard_projection"] == "web/data/"


def test_batch_runner_includes_every_active_task(monkeypatch):
    run_batch = importlib.import_module("src.tasks.run_batch")
    calls = []
    monkeypatch.setattr(run_batch, "run_script", lambda script, args: calls.append(Path(script).name))
    monkeypatch.setattr(
        "sys.argv", ["run_batch.py", "--model", "offline-fixture"]
    )
    run_batch.main()

    task_calls = [name for name in calls if name != "calculate_rationality_stats.py"]
    expected = [Path(item["script"]).name for item in load_manifest()["experiments"]]
    assert task_calls == expected
