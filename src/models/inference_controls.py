"""Pinned provider controls used by canonical benchmark requests."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any


GOOGLE_SAFETY_THRESHOLDS = {
    "harassment": "block_none",
    "hate_speech": "block_none",
    "sexually_explicit": "block_none",
    "dangerous_content": "block_none",
}

PROVIDER_SDK_PACKAGES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google-genai",
}


def openai_reasoning_effort(model_id: str) -> str | None:
    if model_id.startswith("gpt-5.2-pro"):
        return "medium"
    if model_id.startswith("gpt-5.2"):
        return "none"
    if model_id.startswith("o3"):
        return "low"
    if model_id.startswith("gpt-5"):
        return "minimal"
    return None


def google_thinking_control(model_id: str) -> dict[str, Any]:
    if model_id.startswith("gemini-2.5-pro"):
        return {"thinking_budget": 128}
    if model_id.startswith(("gemini-2.5-flash", "gemini-2.5-flash-lite")):
        return {"thinking_budget": 0}
    if model_id.startswith("gemini-3.1-flash-lite"):
        return {"thinking_level": "minimal"}
    raise ValueError(f"no pinned Google thinking control for {model_id!r}")


def recorded_inference_controls(
    provider: str, api_model_id: str
) -> dict[str, Any]:
    sdk_package = PROVIDER_SDK_PACKAGES[provider]
    sdk_controls = {
        "sdk_package": sdk_package,
        "sdk_version": version(sdk_package),
        "sdk_max_retries": 0,
    }
    if provider == "openai":
        effort = openai_reasoning_effort(api_model_id)
        mode = effort if effort is not None else "not_supported"
        return {
            "requested_reasoning_mode": mode,
            "effective_reasoning_mode": mode,
            "provider_options": {
                "reasoning_effort": effort,
                "stateless_request": True,
                **sdk_controls,
            },
        }
    if provider == "anthropic":
        return {
            "requested_reasoning_mode": "disabled",
            "effective_reasoning_mode": "disabled",
            "provider_options": {
                "thinking_parameter": "omitted",
                "thinking_effect": "disabled_for_selected_claude_4_models",
                "stateless_request": True,
                **sdk_controls,
            },
        }
    if provider == "google":
        thinking = google_thinking_control(api_model_id)
        mode = next(iter(thinking.items()))
        rendered_mode = f"{mode[0]}={mode[1]}"
        return {
            "requested_reasoning_mode": rendered_mode,
            "effective_reasoning_mode": rendered_mode,
            "provider_options": {
                **thinking,
                "include_thoughts": False,
                "safety_thresholds": dict(GOOGLE_SAFETY_THRESHOLDS),
                "stateless_request": True,
                **sdk_controls,
            },
        }
    raise ValueError(f"unsupported provider {provider!r}")
