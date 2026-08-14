"""Tests for shared task model loading and response capture."""

import json
import importlib
from types import SimpleNamespace

import pytest

from src.tasks import runtime


ACTIVE_TASKS = [
    "independence",
    "time",
    "dictator",
    "ultimatum",
    "trust_game",
    "stag_hunt",
    "beauty_contest",
    "centipede_game",
    "public_goods",
    "travellers_dilemma",
    "matching_pennies",
]


def write_manifest(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "benchmark-model",
                        "api_model_id": "provider-model-20260811",
                        "provider": "fixture",
                        "status": "active",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_model_loading_uses_the_manifest_provider_endpoint(tmp_path):
    loaded = []

    class FakeInterface:
        pass

    interface = runtime.load_model_interface(
        "benchmark-model",
        loader=lambda model_id: loaded.append(model_id) or FakeInterface(),
        manifest_path=write_manifest(tmp_path),
    )

    assert loaded == ["provider-model-20260811"]
    assert runtime.model_binding_dict(interface) == {
        "model_id": "benchmark-model",
        "api_model_id": "provider-model-20260811",
        "provider": "fixture",
        "status": "active",
        "registered": True,
    }


def test_unregistered_model_identifiers_remain_available_for_offline_fixtures(tmp_path):
    loaded = []
    interface = runtime.load_model_interface(
        "offline-fixture",
        loader=lambda model_id: loaded.append(model_id) or SimpleNamespace(),
        manifest_path=write_manifest(tmp_path),
    )

    assert loaded == ["offline-fixture"]
    assert runtime.model_binding_dict(interface)["registered"] is False


def test_response_capture_logs_full_interaction_and_settings(monkeypatch):
    events = []

    class FakeInterface:
        model_id = "provider-model-20260811"
        econbench_model_binding = runtime.ModelBinding(
            model_id="benchmark-model",
            api_model_id="provider-model-20260811",
            provider="fixture",
            status="active",
            registered=True,
        )

        def generate_response(self, **kwargs):
            assert kwargs == {
                "prompt": "Choose A or B",
                "max_new_tokens": 32,
                "temperature": 0.5,
                "return_logprobs": False,
                "verbose": True,
            }
            return "A", {"prob_a": 0.75}

    monkeypatch.setattr(runtime, "log_event", events.append)
    response = runtime.request_model_response(
        FakeInterface(),
        experiment_id="fixture_game",
        prompt="Choose A or B",
        max_new_tokens=32,
        temperature=0.5,
        verbose=True,
    )

    assert response == "A"
    assert len(events) == 1
    event = events[0]
    assert event["experiment"] == "fixture_game"
    assert event["model"] == "benchmark-model"
    assert event["api_model_id"] == "provider-model-20260811"
    assert event["prompt"] == "Choose A or B"
    assert event["response"] == "A"
    assert event["request"]["temperature"] == 0.5
    assert event["response_valid"] is True
    assert event["error"] is None
    assert event["prompt_sha256"] == runtime.text_sha256("Choose A or B")
    assert event["response_sha256"] == runtime.text_sha256("A")


def test_response_capture_returns_explicit_failure_after_bounded_retries(monkeypatch):
    events = []

    class FailingInterface:
        model_id = "offline-fixture"

        @staticmethod
        def generate_response(**_kwargs):
            raise TimeoutError("provider unavailable")

    monkeypatch.setattr(runtime, "log_event", events.append)

    completion = runtime.request_model_completion(
        FailingInterface(),
        experiment_id="fixture_game",
        prompt="Choose",
        max_new_tokens=8,
        temperature=0.5,
        verbose=False,
        sleeper=lambda _seconds: None,
    )

    assert completion.status == "provider_error"
    assert completion.attempts == 3
    assert len(events) == 1
    assert events[0]["response"] is None
    assert events[0]["response_valid"] is False
    assert events[0]["error"] == {
        "type": "TimeoutError",
        "message": "provider unavailable",
        "retryable": True,
    }


@pytest.mark.parametrize("experiment_id", ACTIVE_TASKS)
def test_every_active_task_routes_responses_through_shared_capture(
    monkeypatch, experiment_id
):
    module = importlib.import_module(f"src.tasks.{experiment_id}")
    interface = object()
    calls = []
    if experiment_id in {"independence", "time"}:
        monkeypatch.setattr(module, "llm", interface)
    monkeypatch.setattr(
        module,
        "request_model_response",
        lambda bound_interface, **kwargs: calls.append((bound_interface, kwargs)) or "A",
    )

    if experiment_id in {"independence", "time"}:
        response = module.generate_response("fixture prompt")
    else:
        response = module.generate_response(interface, "fixture prompt")
    assert response == "A"
    assert calls[0][0] is interface
    assert calls[0][1]["experiment_id"] == experiment_id
    assert calls[0][1]["prompt"] == "fixture prompt"


@pytest.mark.parametrize("result", ["A", (None, None), ("A", None, None)])
def test_response_capture_rejects_invalid_interface_results(monkeypatch, result):
    events = []

    class InvalidInterface:
        model_id = "offline-fixture"

        @staticmethod
        def generate_response(**_kwargs):
            return result

    monkeypatch.setattr(runtime, "log_event", events.append)

    completion = runtime.request_model_completion(
        InvalidInterface(),
        experiment_id="fixture_game",
        prompt="Choose",
        max_new_tokens=8,
        temperature=0.5,
        verbose=False,
        sleeper=lambda _seconds: None,
    )

    assert completion.status == "provider_error"
    assert completion.attempts == 1
    assert events[0]["error"]["type"] == "TypeError"
    assert events[0]["error"]["retryable"] is False
