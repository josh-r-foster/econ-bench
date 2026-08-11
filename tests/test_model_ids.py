"""Tests for canonical model identifier path components."""

import importlib
import json
from pathlib import Path

import pytest

from src.results.model_ids import (
    model_id_from_path_component,
    model_id_to_path_component,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = PROJECT_ROOT / "config" / "models.json"
EXPERIMENTS_PATH = PROJECT_ROOT / "config" / "experiments.json"


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def registered_model_ids():
    return [model["id"] for model in load_json(MODELS_PATH)["models"]]


def test_registered_identifiers_keep_existing_filenames_and_are_unique():
    model_ids = registered_model_ids()
    components = [model_id_to_path_component(model_id) for model_id in model_ids]

    assert components == model_ids
    assert len(components) == len(set(components))


@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("a/b", "~612f62"),
        ("a:b", "~613a62"),
        ("a_b", "a_b"),
        ("A", "~41"),
        (".", "~2e"),
        ("con", "~636f6e"),
        ("con.txt", "~636f6e2e747874"),
        ("model.", "~6d6f64656c2e"),
        ("模型/甲", "~e6a8a1e59e8b2fe794b2"),
    ],
)
def test_path_component_vectors(model_id, expected):
    component = model_id_to_path_component(model_id)
    assert component == expected
    assert model_id_from_path_component(component) == model_id


def test_encoding_removes_old_underscore_collisions():
    model_ids = ["a/b", "a:b", "a_b"]
    old_components = [model_id.replace("/", "_").replace(":", "_") for model_id in model_ids]
    new_components = [model_id_to_path_component(model_id) for model_id in model_ids]

    assert len(set(old_components)) == 1
    assert len(set(new_components)) == len(model_ids)


def test_encoded_components_are_portable_and_case_stable():
    model_ids = ["OpenAI/GPT:4", "space name", "emoji-🙂", "trailing."]
    components = [model_id_to_path_component(model_id) for model_id in model_ids]

    for component in components:
        assert component.startswith("~")
        assert component == component.lower()
        assert not ({"/", "\\", ":", "%", " "} & set(component))


@pytest.mark.parametrize("value", [None, 1, b"gpt-4o"])
def test_encoder_rejects_non_strings(value):
    with pytest.raises(TypeError):
        model_id_to_path_component(value)


def test_encoder_rejects_empty_identifier():
    with pytest.raises(ValueError, match="empty"):
        model_id_to_path_component("")


@pytest.mark.parametrize(
    "component",
    ["", "A", "a/b", "~", "~0", "~zz", "~ff", "~677074"],
)
def test_decoder_rejects_noncanonical_components(component):
    with pytest.raises((TypeError, ValueError, UnicodeDecodeError)):
        model_id_from_path_component(component)


def test_all_task_producers_use_the_shared_helper():
    manifest = load_json(EXPERIMENTS_PATH)
    for experiment in manifest["experiments"]:
        module = importlib.import_module(
            experiment["script"].removesuffix(".py").replace("/", ".")
        )
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert 'replace("/", "_")' not in source, experiment["id"]
        assert "model_id_to_path_component" in source, experiment["id"]


def test_python_consumers_and_browser_pages_use_shared_helpers():
    for path in [
        PROJECT_ROOT / "scripts" / "check_run.py",
        PROJECT_ROOT / "scripts" / "validate_results.py",
        PROJECT_ROOT / "src" / "tools" / "calculate_rationality_stats.py",
    ]:
        source = path.read_text(encoding="utf-8")
        assert 'replace("/", "_")' not in source
        assert "model_id_" in source and "_path_component" in source

    for path in [PROJECT_ROOT / "web" / "index.html", PROJECT_ROOT / "web" / "card.html"]:
        source = path.read_text(encoding="utf-8")
        assert "model_ids.js" in source
        assert "EconBenchModelIds.toPathComponent" in source
        assert "replace(/\\//g" not in source


def test_canonical_location_templates_use_model_key():
    locations = load_json(EXPERIMENTS_PATH)["shared_settings"]["canonical_locations"]
    assert "{model_key}" in locations["raw"]
    assert "{model_key}" in locations["derived"]
    assert all("{model_id}" not in value for value in locations.values())
