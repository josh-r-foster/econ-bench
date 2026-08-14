# EconBench

EconBench is a benchmark for economic choice by language models. Version `1.0.0` contains two elicitation tasks and nine strategic games. The protocol studies stated choices under fixed vignettes. It does not treat those choices as realized interaction between models.

## Canonical scope

The active experiments are Independence, Time, Dictator, Ultimatum, Trust Game, Stag Hunt, Beauty Contest, Centipede Game, Public Goods, Traveller's Dilemma, and Matching Pennies. The experiment and model manifests under `config` define the release design.

Time measures discounting and sensitivity to a front end delay in repeated cross sectional choices. It does not identify within person stability over calendar time. Conditional Ultimatum and Trust responses use a strategy method design.

## Installation

Use Python 3.12 and install the pinned dependencies.

```bash
python -m pip install -r requirements.txt
```

Set the provider keys needed for the selected model in a local `.env` file.

```env
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

## Validation

Run the complete offline gate before any provider request.

```bash
python -m pytest -q
python scripts/validate_protocol.py
python scripts/estimate_release.py --json
python -m compileall -q src scripts tests
git diff --check
git status --short --branch
```

## Running the benchmark

The canonical runner is `scripts/run_benchmark.py`. Native collection requires a clean Git working tree.

Run the full active experiment set for one registered model.

```bash
python scripts/run_benchmark.py --model gpt-4o --run-id pilot-001
```

Run a selected experiment through the same canonical path.

```bash
python scripts/run_benchmark.py --model gpt-4o --run-id pilot-001 --experiments independence
```

Run an offline fixture without provider requests.

```bash
python scripts/run_benchmark.py --model gpt-4o --run-id fixture-001 --fixture
```

The standalone task classes are compatibility and analysis helpers. They do not produce canonical release evidence.

## Results

Canonical raw and derived records are written under `data/releases/1.0.0`. This path is ignored during collection so that checkpoint writes do not change the recorded repository state. Dashboard files under `web/data` are projections that must reproduce from canonical records.

The exact protocol and interpretation limits appear in [docs/protocol.md](docs/protocol.md). The independent review gate appears in [docs/protocol_audit_request.md](docs/protocol_audit_request.md).

## License

See `LICENSE`.
