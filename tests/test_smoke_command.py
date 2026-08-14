"""Offline tests for the opt-in live provider smoke harness."""

import importlib
import subprocess
import sys
from pathlib import Path


def test_smoke_harness_defaults_to_protocol_temperature():
    smoke = importlib.import_module("scripts.smoke_models")
    args = smoke.build_parser().parse_args(["--model", "offline-fixture"])
    assert args.temperature == 0.5


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


def test_smoke_script_imports_project_when_run_directly():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_models.py",
            "--model",
            "offline-fixture",
            "--env-file",
            "missing.env",
        ],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 1
    assert "FAIL offline-fixture ValueError" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr
