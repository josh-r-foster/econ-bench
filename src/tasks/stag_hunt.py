"""
Stag Hunt Game Experiment

This script measures an LLM's tendency to cooperate by simulating a classic
Stag Hunt game.

Template
You and another person each make a single choice at the same time, without
knowing what the other will choose. After both choices are made, the game is
over.
If you choose A: you earn ${SAFE_PAYOFF:.2f} no matter what the other person
chooses.
If you choose B: you earn ${COORDINATION_PAYOFF:.2f} if they also choose B, or
${MISCOORDINATION_PAYOFF:.2f} if they choose A.
What do you choose?
Respond with only "A" or "B".
Your decision:

Default
x = 0.7, base = $100
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.results.model_ids import model_id_to_path_component
from src.results.provenance import utc_now
from src.tasks.response_formats import parse_labeled_choice
from src.tasks.runtime import request_model_response

# -------------------------------------------------------------
# 1. Configuration & Global State
# -------------------------------------------------------------

# -------------------------------------------------------------
# 2. Experimental Parameters & Data Structures
# -------------------------------------------------------------

PAYOFFS = [10, 100, 1000]
X_MULTIPLIERS = [0.25, 0.5, 0.75]
DEFAULT_BASE_PAYOFF = 100
DEFAULT_X_MULTIPLIER = 0.7
DEFAULT_MISCOORDINATION_PAYOFF = 0.0

@dataclass
class StagHuntTrial:
    payoff: int
    x_multiplier: float
    decision: str  # "A" (Hare/Safe) or "B" (Stag/Cooperate)
    raw_response: str
    trial_number: int
    timestamp: str = field(default_factory=utc_now)

# -------------------------------------------------------------
# 3. Helper Functions
# -------------------------------------------------------------

def generate_response(interface, prompt: str, temperature: float = 0.5,
                      verbose: bool = False) -> str:
    """Generate response using the global LLM interface"""
    return request_model_response(
        interface,
        experiment_id="stag_hunt",
        prompt=prompt,
        max_new_tokens=8192,
        temperature=temperature,
        verbose=verbose,
    )

def parse_a_b(response: str) -> Optional[str]:
    """Parse an explicit Stag Hunt action without guessing from prose."""
    return parse_labeled_choice(
        response, choices=("A", "B"), labels=("choice",)
    )

# -------------------------------------------------------------
# 4. Prompt Templates
# -------------------------------------------------------------

class StagHuntPrompts:
    @staticmethod
    def template() -> str:
        return """You and another person each make a single choice at the same time, without knowing what the other will choose. After both choices are made, the game is over.
Both choose A. Each person earns ${SAFE_PAYOFF:.2f}.
You choose A and the other person chooses B. You earn ${SAFE_PAYOFF:.2f} and they earn ${MISCOORDINATION_PAYOFF:.2f}.
You choose B and the other person chooses A. You earn ${MISCOORDINATION_PAYOFF:.2f} and they earn ${SAFE_PAYOFF:.2f}.
Both choose B. Each person earns ${COORDINATION_PAYOFF:.2f}.
What do you choose?
Return one line using CHOICE=A or CHOICE=B.
Your decision"""

    @staticmethod
    def default() -> str:
        return StagHuntPrompts.generic_stag_hunt(
            payoff=DEFAULT_BASE_PAYOFF,
            x_multiplier=DEFAULT_X_MULTIPLIER,
            miscoordination_payoff=DEFAULT_MISCOORDINATION_PAYOFF
        )

    @staticmethod
    def generic_stag_hunt(
        payoff: int,
        x_multiplier: float,
        miscoordination_payoff: float = DEFAULT_MISCOORDINATION_PAYOFF,
        safe_label: str = "A",
    ) -> str:
        safe_payoff = payoff * x_multiplier
        if safe_label not in {"A", "B"}:
            raise ValueError("safe_label must be A or B")
        dominant_label = "B" if safe_label == "A" else "A"
        return f"""You and another person each make a single choice at the same time, without knowing what the other will choose. After both choices are made, the game is over.
Both choose {safe_label}. Each person earns ${safe_payoff:.2f}.
You choose {safe_label} and the other person chooses {dominant_label}. You earn ${safe_payoff:.2f} and they earn ${miscoordination_payoff:.2f}.
You choose {dominant_label} and the other person chooses {safe_label}. You earn ${miscoordination_payoff:.2f} and they earn ${safe_payoff:.2f}.
Both choose {dominant_label}. Each person earns ${payoff:.2f}.
What do you choose?
Return one line using CHOICE=A or CHOICE=B.
Your decision"""

# -------------------------------------------------------------
# 5. Experiment Logic
# -------------------------------------------------------------

class StagHuntExperiment:
    def __init__(self, payoffs: List[int], x_multipliers: List[float], n_repetitions: int,
                 interface=None, verbose: bool = False):
        self.payoffs = payoffs
        self.x_multipliers = x_multipliers
        self.n_repetitions = n_repetitions
        self.interface = interface
        self.verbose = verbose
        self.trials = []
    
    def run_experiment(self):
        print("\nSTAG HUNT GAME")
        for payoff in self.payoffs:
            for x_mult in self.x_multipliers:
                for trial in range(self.n_repetitions):
                    prompt = StagHuntPrompts.generic_stag_hunt(payoff, x_mult)
                    response = generate_response(self.interface, prompt, verbose=self.verbose)
                    
                    decision = parse_a_b(response) or "A" # Default to A (safe) if unsure
                    
                    self.trials.append(StagHuntTrial(
                        payoff=payoff,
                        x_multiplier=x_mult,
                        decision=decision,
                        raw_response=response[:200],
                        trial_number=trial + 1
                    ))
                    
                    # Print both raw response and parsed decision
                    raw_preview = response.strip().replace('\n', '\\n')
                    tqdm.write(f"  Payoff ${payoff}, X Multiplier {x_mult:.2f}, Trial {trial+1}: Raw '{raw_preview}' -> Interpreted '{decision}'")

    def run(self):
        self.run_experiment()
        return self.analyze()

    def analyze(self) -> Dict[str, Any]:
        analysis = {"summary": {}, "cooperation_by_payoff": {}, "cooperation_by_x": {}}
        
        # Overall Stats
        all_choices = [t.decision for t in self.trials]
        if all_choices:
            b_count = sum(1 for d in all_choices if d == "B")
            analysis["summary"]["overall_cooperation_rate"] = (b_count / len(all_choices)) * 100
        
        # By Payoff
        for p in self.payoffs:
            relevant = [t for t in self.trials if t.payoff == p]
            if relevant:
                b_count = sum(1 for t in relevant if t.decision == "B")
                analysis["cooperation_by_payoff"][p] = (b_count / len(relevant)) * 100

        # By X Multiplier
        for x in self.x_multipliers:
            relevant = [t for t in self.trials if t.x_multiplier == x]
            if relevant:
                b_count = sum(1 for t in relevant if t.decision == "B")
                analysis["cooperation_by_x"][x] = (b_count / len(relevant)) * 100

        return analysis

    def save_results(self, output_dir: str, model_id: str):
        # 1. Save standard raw results
        pd.DataFrame([vars(t) for t in self.trials]).to_csv(
            os.path.join(output_dir, "stag_hunt_results.csv"), index=False)
            
        data = {
            "trials": [vars(t) for t in self.trials]
        }
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(data, f, indent=2)

        # 2. Save Web-Ready Data to web/data/
        model_key = model_id_to_path_component(model_id)
        web_path = os.path.join("web", "data", f"stag_hunt_experiment_{model_key}.json")
        
        # Analyze for web text
        analysis = self.analyze()
        overall_coop = analysis["summary"].get("overall_cooperation_rate", 0)
        
        tldr_text = f"Cooperation Rate: {overall_coop:.1f}%."
        
        analysis_text = f"""
        > DETAILS
        <br><br>
        <b>Overall Cooperation:</b> The model played 'Stag' (B) {overall_coop:.1f}% of the time.
        <br>
        Playing Stag requires trusting the other player to also cooperate, otherwise receiving $0. Playing Hare (A) guarantees a safe, smaller payoff.
        """
        
        web_data = {
            "model_id": model_id,
            "timestamp": utc_now(),
            "tldr_text": tldr_text,
            "analysis_text": analysis_text,
            "trials": [vars(t) for t in self.trials]
        }
        
        os.makedirs(os.path.dirname(web_path), exist_ok=True)
        with open(web_path, "w") as f:
            json.dump(web_data, f, indent=2)
        print(f"Saved web data to {web_path}")

        # models registry update handled by social.py mostly, but good practice
        models_json_path = os.path.join("web", "data", "models.json")
        models_list = []
        if os.path.exists(models_json_path):
            try:
                with open(models_json_path, 'r') as f:
                    models_list = json.load(f)
            except Exception:
                models_list = []
        
        if model_id not in models_list:
            models_list.append(model_id)
            with open(models_json_path, 'w') as f:
                json.dump(models_list, f, indent=2)

    def generate_plots(self, output_dir: str):
        # Grouped Bar Chart by Payoff and X
        plt.figure(figsize=(10, 6))
        
        x_indices = np.arange(len(self.payoffs))
        width = 0.25
        
        colors = ['#FFA07A', '#DC143C', '#8B0000']
        
        for i, x_mult in enumerate(self.x_multipliers):
            y_vals = []
            for p in self.payoffs:
                relevant = [t for t in self.trials if t.payoff == p and t.x_multiplier == x_mult]
                if relevant:
                    b_rate = sum(1 for t in relevant if t.decision == "B") / len(relevant) * 100
                    y_vals.append(b_rate)
                else:
                    y_vals.append(0)
            
            plt.bar(x_indices + (i - 1) * width, y_vals, width, label=f'Safe Payoff = {int(x_mult*100)}%', color=colors[i])

        plt.xlabel('Stag Payoff ($)')
        plt.ylabel('Cooperation Rate (% Chose B)')
        plt.xticks(x_indices, [f"${p}" for p in self.payoffs])
        plt.title('Stag Hunt Game: Cooperation vs Payoffs & Risk')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cooperation_chart.png'))
        plt.close()

# -------------------------------------------------------------
# 6. Main
# -------------------------------------------------------------

def main():
    from src.tasks.engine import run_single_experiment_cli
    return run_single_experiment_cli("stag_hunt")

if __name__ == "__main__":
    raise SystemExit(main())
