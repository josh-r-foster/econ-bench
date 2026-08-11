"""Provider registry routing tests."""

from unittest.mock import MagicMock

import pytest

from src.models import registry


ROUTES = [
    ("gpt-4o", "src.models.openai.wrapper"),
    ("o1-preview", "src.models.openai.wrapper"),
    ("o3-mini", "src.models.openai.wrapper"),
    ("claude-3-5-sonnet-20240620", "src.models.anthropic.wrapper"),
    ("gemini-2.5-flash", "src.models.google.wrapper"),
    ("meta-llama/Llama-3.1-70B-Instruct", "src.models.llama_3_1_70b_instruct.wrapper"),
    ("meta-llama/Llama-3.1-8B-Instruct", "src.models.llama_3_1_8b_instruct.wrapper"),
    ("Qwen/Qwen3-8B", "src.models.qwen_3_8b.wrapper"),
]


@pytest.mark.parametrize("model_id, module_path", ROUTES)
def test_registry_routes_every_supported_model_family(monkeypatch, model_id, module_path):
    fake_interface = object()
    fake_module = MagicMock()
    fake_module.LLMInterface.return_value = fake_interface
    importer = MagicMock(return_value=fake_module)
    monkeypatch.setattr(registry.importlib, "import_module", importer)

    assert registry.get_model_interface(model_id) is fake_interface
    importer.assert_called_once_with(module_path)
    fake_module.LLMInterface.assert_called_once_with(model_id=model_id)


def test_registry_rejects_unknown_model(monkeypatch):
    importer = MagicMock()
    monkeypatch.setattr(registry.importlib, "import_module", importer)
    with pytest.raises(ValueError, match="not supported"):
        registry.get_model_interface("unknown/model")
    importer.assert_not_called()
