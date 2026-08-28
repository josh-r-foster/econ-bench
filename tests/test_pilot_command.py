"""Offline tests for the explicit one-model pilot workflow."""

from __future__ import annotations

import importlib


def test_pilot_defaults_to_designated_low_cost_model():
    pilot = importlib.import_module("scripts.run_pilot")
    args = pilot.build_parser().parse_args([])

    assert args.model == "gemini-2.5-flash-lite"
    assert not args.skip_smoke
    assert not args.fixture


def test_live_preflight_requires_provider_credential(monkeypatch, tmp_path):
    pilot = importlib.import_module("scripts.run_pilot")
    monkeypatch.setattr(pilot, "require_clean_repository", lambda _root: "revision")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    result = pilot.main([
        "--run-id",
        "pilot-missing-key",
        "--env-file",
        str(tmp_path / "missing.env"),
        "--preflight-only",
    ])

    assert result == 2


def test_live_pilot_smokes_pinned_endpoint_then_uses_canonical_runner(
    monkeypatch, tmp_path
):
    pilot = importlib.import_module("scripts.run_pilot")
    calls = []
    monkeypatch.setattr(pilot, "require_clean_repository", lambda _root: "revision")
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test-key")
    monkeypatch.setattr(
        pilot,
        "smoke_models_main",
        lambda argv: calls.append(("smoke", argv)) or 0,
    )
    monkeypatch.setattr(
        pilot,
        "run_benchmark_main",
        lambda argv: calls.append(("benchmark", argv)) or 0,
    )

    result = pilot.main([
        "--run-id",
        "pilot-live-test",
        "--env-file",
        str(tmp_path / "missing.env"),
    ])

    assert result == 0
    assert calls == [
        (
            "smoke",
            [
                "--model",
                "gemini-2.5-flash-lite",
                "--max-new-tokens",
                "8",
                "--env-file",
                str(tmp_path / "missing.env"),
            ],
        ),
        (
            "benchmark",
            [
                "--model",
                "gemini-2.5-flash-lite",
                "--run-id",
                "pilot-live-test",
            ],
        ),
    ]


def test_fixture_pilot_skips_credentials_and_smoke(monkeypatch):
    pilot = importlib.import_module("scripts.run_pilot")
    benchmark_calls = []
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(
        pilot,
        "smoke_models_main",
        lambda _argv: (_ for _ in ()).throw(AssertionError("unexpected smoke call")),
    )
    monkeypatch.setattr(
        pilot,
        "run_benchmark_main",
        lambda argv: benchmark_calls.append(argv) or 0,
    )

    result = pilot.main(["--run-id", "pilot-fixture-test", "--fixture"])

    assert result == 0
    assert benchmark_calls == [[
        "--model",
        "gemini-2.5-flash-lite",
        "--run-id",
        "pilot-fixture-test",
        "--fixture",
    ]]


def test_failed_smoke_aborts_before_canonical_collection(monkeypatch, tmp_path):
    pilot = importlib.import_module("scripts.run_pilot")
    monkeypatch.setattr(pilot, "require_clean_repository", lambda _root: "revision")
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test-key")
    monkeypatch.setattr(pilot, "smoke_models_main", lambda _argv: 1)
    monkeypatch.setattr(
        pilot,
        "run_benchmark_main",
        lambda _argv: (_ for _ in ()).throw(AssertionError("unexpected collection")),
    )

    result = pilot.main([
        "--run-id",
        "pilot-smoke-failure",
        "--env-file",
        str(tmp_path / "missing.env"),
    ])

    assert result == 1
