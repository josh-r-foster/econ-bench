"""
Structural interface tests for EconBench model wrappers.

Enforces the LLMInterface contract across all providers WITHOUT
instantiating any wrapper (no API calls, no GPU required).

Run with:
    pytest tests/test_model_interfaces.py -v
"""

import importlib
import inspect
import sys
from pathlib import Path
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# All wrapper module paths to check
# ---------------------------------------------------------------------------

WRAPPER_MODULES = [
    "src.models.openai.wrapper",
    "src.models.anthropic.wrapper",
    "src.models.google.wrapper",
    "src.models.llama_3_1_70b_instruct.wrapper",
    "src.models.llama_3_1_8b_instruct.wrapper",
    "src.models.qwen_3_8b.wrapper",
]

# Required generate_response parameters (name → required or optional)
GENERATE_RESPONSE_PARAMS = {
    "prompt": True,
    "max_new_tokens": False,
    "temperature": False,
    "return_logprobs": False,
    "verbose": False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_module_safely(module_path: str):
    """Import a wrapper module, mocking heavy optional imports so they don't fail."""
    # Provide lightweight stubs for GPU/API packages that may not be installed
    stubs = ["torch", "transformers", "anthropic", "openai", "google", "google.genai",
             "google.genai.types", "bitsandbytes", "accelerate"]
    mock_modules = {}
    for name in stubs:
        if name not in sys.modules:
            from unittest.mock import MagicMock
            mock_modules[name] = MagicMock()

    original = {}
    for name, mock in mock_modules.items():
        original[name] = sys.modules.get(name)
        sys.modules[name] = mock

    try:
        # Force reimport so stubs take effect
        if module_path in sys.modules:
            del sys.modules[module_path]
        module = importlib.import_module(module_path)
    finally:
        for name, orig in original.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig

    return module


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_path", WRAPPER_MODULES)
def test_wrapper_has_llm_interface_class(module_path):
    mod = _import_module_safely(module_path)
    assert hasattr(mod, "LLMInterface"), (
        f"{module_path} must define a class named 'LLMInterface'"
    )
    assert inspect.isclass(mod.LLMInterface), (
        f"{module_path}.LLMInterface must be a class"
    )


@pytest.mark.parametrize("module_path", WRAPPER_MODULES)
def test_generate_response_exists_and_callable(module_path):
    mod = _import_module_safely(module_path)
    cls = mod.LLMInterface
    assert hasattr(cls, "generate_response"), (
        f"{module_path}.LLMInterface must have a 'generate_response' method"
    )
    assert callable(cls.generate_response), (
        f"{module_path}.LLMInterface.generate_response must be callable"
    )


@pytest.mark.parametrize("module_path", WRAPPER_MODULES)
def test_generate_response_signature(module_path):
    mod = _import_module_safely(module_path)
    sig = inspect.signature(mod.LLMInterface.generate_response)
    params = sig.parameters
    # Remove 'self'
    param_names = [p for p in params if p != "self"]

    for name, required in GENERATE_RESPONSE_PARAMS.items():
        assert name in param_names, (
            f"{module_path}.LLMInterface.generate_response is missing parameter '{name}'"
        )
        if required:
            assert params[name].default is inspect.Parameter.empty, (
                f"Parameter '{name}' must be required (no default) in {module_path}"
            )


@pytest.mark.parametrize("module_path", WRAPPER_MODULES)
def test_parse_ab_choice_exists_and_callable(module_path):
    mod = _import_module_safely(module_path)
    cls = mod.LLMInterface
    assert hasattr(cls, "parse_ab_choice"), (
        f"{module_path}.LLMInterface must have a 'parse_ab_choice' method"
    )
    assert callable(cls.parse_ab_choice), (
        f"{module_path}.LLMInterface.parse_ab_choice must be callable"
    )


@pytest.mark.parametrize("module_path", WRAPPER_MODULES)
def test_parse_ab_choice_signature(module_path):
    mod = _import_module_safely(module_path)
    sig = inspect.signature(mod.LLMInterface.parse_ab_choice)
    params = [p for p in sig.parameters if p != "self"]
    assert "response" in params, (
        f"{module_path}.LLMInterface.parse_ab_choice must accept a 'response' parameter"
    )


# ---------------------------------------------------------------------------
# Registry routing test
# ---------------------------------------------------------------------------

def test_registry_routes_known_prefixes():
    """registry.get_model_interface should not raise ValueError for known prefixes."""
    from unittest.mock import patch, MagicMock

    fake_interface = MagicMock()

    with patch("importlib.import_module") as mock_import:
        mock_module = MagicMock()
        mock_module.LLMInterface.return_value = fake_interface
        mock_import.return_value = mock_module

        registry = importlib.import_module("src.models.registry")

        test_ids = ["gpt-4o", "o1-preview", "o3-mini", "claude-3-5-sonnet-20240620",
                    "gemini-2.0-flash", "gemini-1.5-pro"]
        for model_id in test_ids:
            try:
                registry.get_model_interface(model_id)
            except ValueError as e:
                pytest.fail(f"registry raised ValueError for '{model_id}': {e}")
