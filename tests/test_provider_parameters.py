"""Offline checks for provider specific request parameters."""

from types import SimpleNamespace

import pytest

from src.models.anthropic import wrapper as anthropic_wrapper
from src.models.openai import wrapper as openai_wrapper
from src.models.inference_controls import (
    google_thinking_control,
    openai_reasoning_effort,
    recorded_inference_controls,
)
from src.tasks import engine


@pytest.fixture(autouse=True)
def disable_wrapper_logging(monkeypatch):
    monkeypatch.setattr(anthropic_wrapper, "log_model_call", lambda **_kwargs: None)
    monkeypatch.setattr(openai_wrapper, "log_model_call", lambda **_kwargs: None)


class AnthropicClient:
    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Success")],
            usage=SimpleNamespace(input_tokens=2, output_tokens=1),
        )


class ChatCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="Success"), logprobs=None
            )],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
        )


class Responses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text="Success",
            usage=SimpleNamespace(input_tokens=2, output_tokens=1),
        )


def anthropic_interface(model_id):
    interface = object.__new__(anthropic_wrapper.LLMInterface)
    interface.model_id = model_id
    interface.client = AnthropicClient()
    return interface


def openai_interface(model_id, api_mode):
    interface = object.__new__(openai_wrapper.LLMInterface)
    interface.model_id = model_id
    interface._api_mode = api_mode
    interface.client = SimpleNamespace(
        chat=SimpleNamespace(completions=ChatCompletions()),
        responses=Responses(),
    )
    return interface


def test_anthropic_omits_unsupported_temperature():
    interface = anthropic_interface("claude-opus-4-7")
    interface.generate_response("Reply", max_new_tokens=8, temperature=0.5)
    request = interface.client.calls[0]
    assert request["max_tokens"] == 8
    assert "temperature" not in request


def test_anthropic_sends_supported_temperature():
    interface = anthropic_interface("claude-haiku-4-5-20251001")
    interface.generate_response("Reply", max_new_tokens=8, temperature=0.5)
    assert interface.client.calls[0]["temperature"] == 0.5


def test_openai_reasoning_models_honor_output_limit():
    interface = openai_interface("gpt-5-nano-2025-08-07", "chat")
    interface.generate_response("Reply", max_new_tokens=321, temperature=0.5)
    request = interface.client.chat.completions.calls[0]
    assert request["max_completion_tokens"] == 321
    assert "temperature" not in request
    assert request["reasoning_effort"] == "minimal"


def test_openai_responses_models_honor_output_limit():
    interface = openai_interface("gpt-5.2-pro-2025-12-11", "responses")
    interface.generate_response("Reply", max_new_tokens=321, temperature=0.5)
    request = interface.client.responses.calls[0]
    assert request["max_output_tokens"] == 321
    assert "temperature" not in request
    assert request["reasoning"] == {"effort": "medium"}


def test_provider_wrappers_preserve_raw_whitespace():
    anthropic = anthropic_interface("claude-haiku-4-5-20251001")
    anthropic.client.create = lambda **_kwargs: SimpleNamespace(
        content=[SimpleNamespace(type="text", text="  DECISION=ACCEPT\n")],
        usage=None,
    )
    assert anthropic.generate_response("Reply")[0] == "  DECISION=ACCEPT\n"

    openai = openai_interface("gpt-4o-2024-11-20", "chat")
    openai.client.chat.completions.create = lambda **_kwargs: SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="  CHOICE=A\n"), logprobs=None
        )],
        usage=None,
    )
    assert openai.generate_response("Reply")[0] == "  CHOICE=A\n"


def test_reasoning_controls_are_fixed_and_recordable():
    assert openai_reasoning_effort("gpt-5.2-2025-12-11") == "none"
    assert google_thinking_control("gemini-2.5-flash") == {
        "thinking_budget": 0
    }
    assert google_thinking_control("gemini-2.5-pro") == {
        "thinking_budget": 128
    }
    assert google_thinking_control("gemini-3.1-flash-lite") == {
        "thinking_level": "minimal"
    }
    controls = recorded_inference_controls(
        "google", "gemini-2.5-flash-lite"
    )
    assert controls["effective_reasoning_mode"] == "thinking_budget=0"
    assert controls["provider_options"]["safety_thresholds"]
    assert controls["provider_options"]["sdk_package"] == "google-genai"
    assert controls["provider_options"]["sdk_version"]
    assert controls["provider_options"]["sdk_max_retries"] == 0


def test_opus_metadata_records_null_effective_temperature(monkeypatch):
    monkeypatch.setattr(engine, "git_provenance", lambda _root: ("a" * 40, False))
    monkeypatch.setattr(
        engine,
        "model_config",
        lambda _model_id: {
            "provider": "anthropic",
            "api_model_id": "claude-opus-4-7",
            "status": "active",
        },
    )
    metadata = engine._metadata(
        "claude-opus-4-7",
        {
            "id": "dictator",
            "family": "strategic_game",
            "settings": {"pool_amounts": [10]},
            "temperature": 0.5,
            "max_output_tokens": 8,
            "response_parser": "parse_dollar_amount",
        },
        "fixture-run",
        "2026-08-11T00:00:00Z",
        runner="fixture",
        project_root=engine.PROJECT_ROOT,
    )
    parameters = metadata["model"]["parameters"]
    assert parameters["requested_temperature"] == 0.5
    assert parameters["effective_temperature"] is None
