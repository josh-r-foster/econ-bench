"""Keep published browser assets and catalog consistent without provider calls."""
import json
from pathlib import Path

import pytest

from scripts.build_web_catalog import build_catalog

ROOT = Path(__file__).resolve().parents[1]


def test_committed_catalog_matches_published_files():
    assert json.loads((ROOT / 'web/catalog.json').read_text()) == build_catalog(ROOT / 'web')


def test_catalog_excludes_empty_models_and_preserves_encoded_ids(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'models.json').write_text(json.dumps(['empty', 'provider/model']))
    filename = 'beauty_contest_experiment_~70726f76696465722f6d6f64656c.json'
    (data / filename).write_text(json.dumps({'model_id': 'provider/model', 'trials': [{'decision': 0}]}))
    assert build_catalog(tmp_path) == {'models': [{'id': 'provider/model', 'files': {'beauty_contest': filename}}]}


def test_catalog_rejects_misattributed_results(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'models.json').write_text('["model"]')
    (data / 'time_experiment_model.json').write_text('{"model_id":"another-model"}')
    with pytest.raises(ValueError, match='does not match'):
        build_catalog(tmp_path)
