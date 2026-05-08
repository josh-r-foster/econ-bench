# EconBench — Agent Guide

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run a single experiment
python src/tasks/independence.py --model gpt-4o
python src/tasks/time.py --model gpt-4o
python src/tasks/social.py --model gpt-4o

# Run all three core experiments for one model
python src/tasks/run_batch.py --model gpt-4o

# Run any of the additional game experiments
python src/tasks/beauty_contest.py --model gpt-4o
python src/tasks/stag_hunt.py --model gpt-4o

# Recalculate rationality scores after new runs
python src/tools/calculate_rationality_stats.py

# Validate results completeness across all registered models
python scripts/validate_results.py

# Check a completed run's quality (four-layer eval)
python scripts/check_run.py --model gpt-4o

# View the dashboard (open in browser after starting server)
cd web && python -m http.server 8000
```

## Project Structure

```
src/
  models/           # LLM provider wrappers (one subdir per provider)
    registry.py     # Factory: routes model ID → correct LLMInterface
    logger.py       # JSONL trace logging (written per generate_response call)
    openai/         # GPT-4o, o1, o3, gpt-5 family
    anthropic/      # Claude family
    google/         # Gemini family
    llama_*/        # Local Llama (requires GPU + HuggingFace token)
    qwen_*/         # Local Qwen (requires GPU)
  tasks/            # One script per experiment
    run_batch.py    # Orchestrates independence + time + social
    independence.py # Marschak-Machina Triangle (risk/rationality)
    time.py         # Intertemporal discount rate elicitation
    social.py       # Dictator and Ultimatum games
    beauty_contest.py, stag_hunt.py, ...  # Additional games
  tools/
    calculate_rationality_stats.py  # Aggregates scores → web/data/
scripts/
  validate_results.py  # Schema and completeness check for all models
  check_run.py         # Four-layer quality check on a single model's run
  run_benchmark.py     # Legacy batch runner
  update_leaderboard.py
web/
  index.html       # Main dashboard
  card.html        # Per-model results card
  data/            # JSON files consumed by the dashboard
    models.json    # Registered model list (source of truth for validate)
data/
  results/
    independence/{model}/  # mm_triangle_results.json, .csv, .txt, .png
    time/{model}/
    social/{model}/
    runs/          # JSONL trace logs (one file per session)
```

## Adding a New Model Wrapper

1. Create `src/models/{provider}/wrapper.py` implementing `LLMInterface`:

```python
from typing import Optional, Dict, Tuple
try:
    from ..logger import log_event
except ImportError:
    def log_event(_): pass

class LLMInterface:
    def __init__(self, model_id: str, device: str = "auto"):
        self.model_id = model_id
        # initialize client here

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.01,
        return_logprobs: bool = False,
        verbose: bool = False,
    ) -> Tuple[str, Optional[Dict]]:
        # call API, then:
        log_event({
            "event": "model_call",
            "model": self.model_id,
            "prompt_chars": len(prompt),
            "valid": bool(content),
        })
        return content, logprob_dict

    def parse_ab_choice(self, response: str) -> Optional[str]:
        # return "A", "B", or None
        ...
```

2. Register in `src/models/registry.py`:
   - For prefix-matched providers (e.g., `my-provider-`), add an `if` branch.
   - For exact model IDs, add to the `MODEL_MAP` dict.

3. Add the model ID to `web/data/models.json` after running experiments.

## Adding a New Experiment

1. Create `src/tasks/{experiment_name}.py`. Required structure:
   - Accept `--model` CLI argument (use `argparse`)
   - Load model via `from src.models.registry import get_model_interface`
   - Save detailed results to `data/results/{experiment}/{model}/`
   - Save dashboard-ready JSON to `web/data/{experiment}_experiment_{model}.json`

2. Optionally wire into `run_batch.py` if it should be part of the core suite.

## Code Conventions

- **Temperature**: always `0.01` (near-deterministic) for experimental validity
- **Bisection iterations**: `10` by default across all elicitation tasks
- **Dual storage**: every task writes to both `data/results/` (detailed) and `web/data/` (dashboard JSON)
- **Result schema**: result JSON must have a top-level `"results"` array; rationality JSON must have a `"metrics"` object
- **JSONL traces**: automatically written to `data/results/runs/` for every `generate_response` call — used by `check_run.py`
- **Error handling**: on API failure, wrappers return `("", None)` — tasks count empty responses as invalid

## Environment Variables

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
ECONBENCH_LOG_DIR=data/results/runs   # override trace log location; set to "none" to disable
ECONBENCH_EXPERIMENT=independence     # optionally tag traces with experiment name
```

Copy `.env.example` to `.env` and fill in your keys.
