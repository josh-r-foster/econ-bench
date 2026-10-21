#!/usr/bin/env python3
"""Index published dashboard files; no provider calls or result modification."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = (
    'time', 'independence', 'dictator', 'ultimatum', 'trust_game', 'stag_hunt',
    'beauty_contest', 'centipede_game', 'public_goods', 'travellers_dilemma',
    'matching_pennies',
)


def model_key(model):
    # Keep filename encoding identical to the canonical Python and browser helpers.
    from src.results.model_ids import model_id_to_path_component
    return model_id_to_path_component(model)


def build_catalog(web_root):
    models = json.loads((web_root / 'data/models.json').read_text())
    catalog = {'models': []}
    for model in models:
        key = model_key(model)
        files = {}
        for experiment in (*EXPERIMENTS, 'social', 'rationality'):
            if experiment == 'rationality':
                filename = f'{key}_rationality.json'
            else:
                prefix = 'independence_results' if experiment == 'independence' else f'{experiment}_experiment'
                filename = f'{prefix}_{key}.json'
            path = web_root / 'data' / filename
            if path.is_file():
                data = json.loads(path.read_text())
                owner = data.get('model_id', data.get('model'))
                if owner is not None and owner != model:
                    raise ValueError(f'{path}: result model does not match filename')
                files[experiment] = filename
        if files:
            catalog['models'].append({'id': model, 'files': files})
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = ROOT / 'web/catalog.json'
    content = json.dumps(build_catalog(ROOT / 'web'), indent=2) + '\n'
    if args.check:
        if not path.exists() or path.read_text() != content:
            parser.exit(1, 'Website catalog is stale. Run python scripts/build_web_catalog.py\n')
        print('Website catalog is current.')
    else:
        path.write_text(content)
        print(path)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    main()
