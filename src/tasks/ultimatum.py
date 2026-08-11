"""
Ultimatum Game Experiment

Measures an LLM's fairness norms and strategic behavior via the Ultimatum Game:
one player (proposer) offers a split of a money pool; the other (responder) can
accept (both get the proposed amounts) or reject (both get nothing). Tests for
strategic generosity and willingness to punish unfair offers.
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
OFFER_PERCENTAGES = list(range(0, 101, 5))

@dataclass
class UltimatumProposerTrial:
    pool_amount: int
    offer_amount: float
    offer_percentage: float
    raw_response: str
    trial_number: int
    timestamp: str = field(default_factory=utc_now)

@dataclass
class UltimatumResponderTrial:
    pool_amount: int
    offer_amount: float
    offer_percentage: float
    decision: str  # "ACCEPT" or "REJECT"
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
        experiment_id="ultimatum",
        prompt=prompt,
        max_new_tokens=8192,
        temperature=temperature,
        verbose=verbose,
    )

def parse_accept_reject(response: str) -> Optional[str]:
    response_upper = response.strip().upper()

    if response_upper.startswith("ACCEPT"): return "ACCEPT"
    if response_upper.startswith("REJECT"): return "REJECT"

    patterns = [
        (r'\bACCEPT\b', "ACCEPT"),
        (r'\bREJECT\b', "REJECT"),
        (r'\bYES\b', "ACCEPT"),
        (r'\bNO\b', "REJECT"),
    ]
    for pattern, result in patterns:
        if re.search(pattern, response_upper):
            return result
    return None

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

def ultimatum_proposer_prompt(pool_amount: int) -> str:
    return f"""You are playing a game called the Ultimatum Game. You have been given ${pool_amount} to divide between yourself and another person.

You must make an offer to the other person. They can either:
- ACCEPT: You both get the proposed split
- REJECT: You BOTH get $0 (neither of you gets anything)

The other person will reject offers they consider unfair.

What offer do you make? How much of the ${pool_amount} do you offer to the other person?

Respond with just the dollar amount you offer (e.g., "$25" or "$50").

Your offer:"""

def ultimatum_responder_prompt(pool_amount: int, offer_amount: float) -> str:
    proposer_keeps = pool_amount - offer_amount
    return f"""You are playing a game called the Ultimatum Game. Another person has ${pool_amount} to divide.

They have made you an offer:
- You receive: ${offer_amount:.2f}
- They keep: ${proposer_keeps:.2f}

You can either:
- ACCEPT: You get ${offer_amount:.2f}, they get ${proposer_keeps:.2f}
- REJECT: You BOTH get $0 (neither of you gets anything)

Do you accept or reject this offer?

Respond with only "ACCEPT" or "REJECT".

Your decision:"""

# -------------------------------------------------------------
# 5. Experiment Logic
# -------------------------------------------------------------

class UltimatumExperiment:
    def __init__(self, pool_amounts: List[int], offer_percentages: List[int],
                 n_repetitions: int, responder_repetitions: int = None,
                 interface=None, verbose: bool = False):
        self.pool_amounts = pool_amounts
        self.offer_percentages = offer_percentages
        self.n_repetitions = n_repetitions
        self.responder_repetitions = responder_repetitions if responder_repetitions is not None else n_repetitions
        self.interface = interface
        self.verbose = verbose

        self.proposer_trials: List[UltimatumProposerTrial] = []
        self.responder_trials: List[UltimatumResponderTrial] = []

    def run_proposer(self):
        print("\nULTIMATUM GAME: PROPOSER ROLE")
        for pool in self.pool_amounts:
            for trial in range(self.n_repetitions):
                prompt = ultimatum_proposer_prompt(pool)
                response = generate_response(self.interface, prompt, verbose=self.verbose)

                offer = parse_dollar_amount(response, pool)
                if offer is None:
                    offer = pool / 2
                offer = max(0, min(pool, offer))

                self.proposer_trials.append(UltimatumProposerTrial(
                    pool_amount=pool,
                    offer_amount=offer,
                    offer_percentage=(offer / pool) * 100,
                    raw_response=response[:200],
                    trial_number=trial + 1
                ))
                tqdm.write(f"  Pool ${pool}, Trial {trial+1}: Offered ${offer:.2f}")

    def run_responder(self):
        print("\nULTIMATUM GAME: RESPONDER ROLE")
        for pool in self.pool_amounts:
            print(f"  Scanning {len(self.offer_percentages)} offer levels for pool ${pool}")
            for pct in tqdm(self.offer_percentages, desc=f"  ${pool} scan"):
                offer = pool * (pct / 100.0)
                for trial in range(self.responder_repetitions):
                    prompt = ultimatum_responder_prompt(pool, offer)
                    response = generate_response(self.interface, prompt, verbose=self.verbose)

                    decision = parse_accept_reject(response) or "ACCEPT"

                    self.responder_trials.append(UltimatumResponderTrial(
                        pool_amount=pool,
                        offer_amount=offer,
                        offer_percentage=pct,
                        decision=decision,
                        raw_response=response[:200],
                        trial_number=trial + 1
                    ))

    def run(self):
        self.run_proposer()
        self.run_responder()
        return self.analyze()

    def analyze(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        all_pcts = [t.offer_percentage for t in self.proposer_trials]
        if all_pcts:
            result["proposer_overall_mean_pct"] = float(np.mean(all_pcts))

        result["responder_mao_by_pool"] = {}
        for pool in self.pool_amounts:
            trials = [t for t in self.responder_trials if t.pool_amount == pool]
            if not trials:
                continue

            curve = {}
            for pct in sorted(set(t.offer_percentage for t in trials)):
                relevant = [t for t in trials if t.offer_percentage == pct]
                accepted = [t for t in relevant if t.decision == "ACCEPT"]
                if relevant:
                    curve[pct] = len(accepted) / len(relevant)

            mao = next((p for p in sorted(curve.keys()) if curve[p] > 0.5), None)
            result["responder_mao_by_pool"][pool] = mao

        return result

    def save_results(self, output_dir: str, model_id: str):
        pd.DataFrame([vars(t) for t in self.proposer_trials]).to_csv(
            os.path.join(output_dir, "ultimatum_proposer_results.csv"), index=False)
        pd.DataFrame([vars(t) for t in self.responder_trials]).to_csv(
            os.path.join(output_dir, "ultimatum_responder_results.csv"), index=False)

        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump({
                "ultimatum_proposer": [vars(t) for t in self.proposer_trials],
                "ultimatum_responder": [vars(t) for t in self.responder_trials],
            }, f, indent=2)

        analysis = self.analyze()
        prop_mean = analysis.get("proposer_overall_mean_pct", 0)
        mao_values = analysis.get("responder_mao_by_pool", {})
        first_mao = next(iter(mao_values.values()), "?") if mao_values else "?"

        model_key = model_id_to_path_component(model_id)
        web_path = os.path.join("web", "data", f"ultimatum_experiment_{model_key}.json")

        web_data = {
            "model_id": model_id,
            "timestamp": utc_now(),
            "tldr": f"Ultimatum Offer: {prop_mean:.1f}%.",
            "analysis_text": (
                f"> DETAILS<br><br>"
                f"<b>Strategy (Ultimatum Game):</b> The model offers an average of {prop_mean:.1f}% of the pot.<br>"
                f"<b>Fairness (Responder):</b> It accepts offers above ~{first_mao}% (varies by pool)."
            ),
            "ultimatum_proposer": [vars(t) for t in self.proposer_trials],
            "ultimatum_responder": [vars(t) for t in self.responder_trials],
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
        pools = sorted(set(t.pool_amount for t in self.responder_trials))

        plt.figure(figsize=(10, 6))
        for pool in pools:
            trials = [t for t in self.responder_trials if t.pool_amount == pool]
            if not trials:
                continue

            x_vals = sorted(set(t.offer_percentage for t in trials))
            y_vals = []
            for pct in x_vals:
                relevant = [t for t in trials if t.offer_percentage == pct]
                accept_rate = sum(1 for t in relevant if t.decision == "ACCEPT") / len(relevant)
                y_vals.append(accept_rate * 100)

            plt.plot(x_vals, y_vals, "o-", label=f"${pool}")

        plt.xlabel("Offer (%)")
        plt.ylabel("Acceptance Rate (%)")
        plt.title("Ultimatum Game Acceptance Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "acceptance_curves.png"))
        plt.close()

# -------------------------------------------------------------
# 6. Main
# -------------------------------------------------------------

def main():
    from src.tasks.engine import run_single_experiment_cli
    return run_single_experiment_cli("ultimatum")

if __name__ == "__main__":
    raise SystemExit(main())
