"""Offline tests for the opt-in live provider smoke harness."""

import importlib


def test_smoke_harness_uses_registry_without_running_at_import(monkeypatch):
    smoke = importlib.import_module("scripts.smoke_models")
    registry = importlib.import_module("src.models.registry")

    class FakeInterface:
        @staticmethod
        def generate_response(**kwargs):
            return "Success", None

    monkeypatch.setattr(registry, "get_model_interface", lambda model_id: FakeInterface())
    result = smoke.main(["--model", "offline-fixture", "--env-file", "missing.env"])
    assert result == 0


def test_smoke_harness_returns_failure_for_provider_error(monkeypatch):
    smoke = importlib.import_module("scripts.smoke_models")
    registry = importlib.import_module("src.models.registry")

    def fail(model_id):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(registry, "get_model_interface", fail)
    result = smoke.main(["--model", "offline-fixture", "--env-file", "missing.env"])
    assert result == 1
