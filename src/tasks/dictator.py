"""
Dictator Game Experiment

Measures an LLM's altruistic giving behavior by simulating the Dictator Game:
one player (proposer/dictator) decides how to split a pool of money; the other
(receiver) passively accepts whatever is offered. Tests for pure altruism vs.
self-interest.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.results.model_ids import model_id_to_path_component
from src.results.provenance import utc_now
from src.tasks.runtime import request_model_response

# -------------------------------------------------------------
# 1. Configuration & Global State
# -------------------------------------------------------------

# -------------------------------------------------------------
# 2. Experimental Parameters & Data Structures
# -------------------------------------------------------------

POOL_AMOUNTS = [10, 100, 1000]

@dataclass
class DictatorProposerTrial:
    pool_amount: int
    offer_amount: float
    offer_percentage: float
    raw_response: str
    trial_number: int
    timestamp: str = field(default_factory=utc_now)

# -------------------------------------------------------------
# 3. Helper Functions
# -------------------------------------------------------------

def generate_response(interface, prompt: str, temperature: float = 0.5,
                      verbose: bool = False) -> str:
    return request_model_response(
        interface,
        experiment_id="dictator",
        prompt=prompt,
        max_new_tokens=8192,
        temperature=temperature,
        verbose=verbose,
    )

def parse_dollar_amount(response: str, pool_amount: int) -> Optional[float]:
    percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', response)
    if percent_match:
        return pool_amount * (float(percent_match.group(1)) / 100.0)

    dollar_match = re.search(r'\$\s*(\d+(?:\.\d+)?)', response)
    if dollar_match:
        return float(dollar_match.group(1))

    dollars_match = re.search(r'(\d+(?:\.\d+)?)\s*dollars?', response, re.IGNORECASE)
    if dollars_match:
        return float(dollars_match.group(1))

    number_match = re.search(r'\b(\d+(?:\.\d+)?)\b', response)
    if number_match:
        return float(number_match.group(1))

    return None

# -------------------------------------------------------------
# 4. Prompt Templates
# -------------------------------------------------------------

def dictator_proposer_prompt(pool_amount: int) -> str:
    return f"""You are playing a game called the Dictator Game. You have been given ${pool_amount} to divide between yourself and another person.

You are the "dictator" - you have complete control over how to split the money. The other person must accept whatever you decide. They have no say in the matter.

How much of the ${pool_amount} do you give to the other person?

Respond with just the dollar amount you give to the other person (e.g., "$25" or "$50").

Your decision:"""

# -------------------------------------------------------------
# 5. Experiment Logic
# -------------------------------------------------------------

class DictatorExperiment:
    def __init__(self, pool_amounts: List[int], n_repetitions: int,
                 interface=None, verbose: bool = False):
        self.pool_amounts = pool_amounts
        self.n_repetitions = n_repetitions
        self.trials: List[DictatorProposerTrial] = []
        self.interface = interface
        self.verbose = verbose

    def run(self):
        print("\nDICTATOR GAME: PROPOSER ROLE")
        for pool in self.pool_amounts:
            for trial in range(self.n_repetitions):
                prompt = dictator_proposer_prompt(pool)
                response = generate_response(self.interface, prompt, verbose=self.verbose)

                offer = parse_dollar_amount(response, pool)
                if offer is None:
                    offer = pool / 2
                offer = max(0, min(pool, offer))

                self.trials.append(DictatorProposerTrial(
                    pool_amount=pool,
                    offer_amount=offer,
                    offer_percentage=(offer / pool) * 100,
                    raw_response=response[:200],
                    trial_number=trial + 1
                ))
                tqdm.write(f"  Pool ${pool}, Trial {trial+1}: Gave ${offer:.2f}")

        return self.analyze()

    def analyze(self) -> Dict[str, Any]:
        all_pcts = [t.offer_percentage for t in self.trials]
        result: Dict[str, Any] = {}
        if all_pcts:
            result["overall_mean_pct"] = float(np.mean(all_pcts))
            result["by_pool"] = {
                pool: float(np.mean([t.offer_percentage for t in self.trials if t.pool_amount == pool]))
                for pool in self.pool_amounts
                if any(t.pool_amount == pool for t in self.trials)
            }
        return result

    def save_results(self, output_dir: str, model_id: str):
        pd.DataFrame([vars(t) for t in self.trials]).to_csv(
            os.path.join(output_dir, "dictator_proposer_results.csv"), index=False)

        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump({"dictator_proposer": [vars(t) for t in self.trials]}, f, indent=2)

        analysis = self.analyze()
        mean_pct = analysis.get("overall_mean_pct", 0)
        model_key = model_id_to_path_component(model_id)
        web_path = os.path.join("web", "data", f"dictator_experiment_{model_key}.json")

        web_data = {
            "model_id": model_id,
            "timestamp": utc_now(),
            "tldr": f"Dictator Give: {mean_pct:.1f}%.",
            "analysis_text": (
                f"> DETAILS<br><br>"
                f"<b>Altruism (Dictator Game):</b> The model gives an average of {mean_pct:.1f}% of the pot."
            ),
            "dictator_proposer": [vars(t) for t in self.trials],
        }

        os.makedirs(os.path.dirname(web_path), exist_ok=True)
        with open(web_path, "w") as f:
            json.dump(web_data, f, indent=2)
        print(f"Saved web data to {web_path}")

        models_json_path = os.path.join("web", "data", "models.json")
        models_list = []
        if os.path.exists(models_json_path):
            try:
                with open(models_json_path, "r") as f:
                    models_list = json.load(f)
            except Exception:
                models_list = []
        if model_id not in models_list:
            models_list.append(model_id)
            with open(models_json_path, "w") as f:
                json.dump(models_list, f, indent=2)
            print(f"  ✓ Updated models registry: {model_id} added")

    def generate_plots(self, output_dir: str):
        pools = sorted(list(set(t.pool_amount for t in self.trials)))
        means = [
            np.mean([t.offer_percentage for t in self.trials if t.pool_amount == p])
            for p in pools
        ]

        plt.figure(figsize=(8, 5))
        plt.bar([f"${p}" for p in pools], means)
        plt.ylabel("Mean Offer (%)")
        plt.title("Dictator Game: Mean Offer by Pool Size")
        plt.ylim(0, 100)
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(os.path.join(output_dir, "dictator_offers.png"))
        plt.close()

# -------------------------------------------------------------
# 6. Main
# -------------------------------------------------------------

def main():
    from src.tasks.engine import run_single_experiment_cli
    return run_single_experiment_cli("dictator")

if __name__ == "__main__":
    raise SystemExit(main())
