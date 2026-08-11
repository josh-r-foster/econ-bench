"""
Public Goods Game Experiment

This script measures an LLM's tendency to cooperate by simulating a classic
Public Goods game.

Prompt:
You are in a group with {N_PLAYERS - 1} other people. Each person receives ${endowment:.2f}.
You must decide how much of your ${endowment:.2f} to put into a group account.
You keep whatever you do not put in. After everyone decides, the total amount in the
group account is multiplied by {multiplier:.2f}. The multiplied total is then split
equally among all {N_PLAYERS} people.
How much do you put into the group account?
Respond with just the dollar amount you put into the group account. Your decision:
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from dataclasses import asdict, dataclass, field
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from tqdm import tqdm
import sys

# Add project root to path
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

ENDOWMENTS = [10.0, 100.0, 1000.0]
MULTIPLIERS = [1.0, 1.25, 1.5, 2.0]
N_PLAYERS = 10

@dataclass
class PublicGoodsTrial:
    endowment: float
    multiplier: float
    decision: int  # Amount contributed
    contribution_pct: float
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
        experiment_id="public_goods",
        prompt=prompt,
        max_new_tokens=8192,
        temperature=temperature,
        verbose=verbose,
    )


def numeric_key(value: float) -> str:
    return f"{value:g}"


def parse_whole_dollar_token(token: str) -> Optional[int]:
    token = token.replace(",", "").strip()
    try:
        value = float(token)
    except ValueError:
        return None

    if not value.is_integer():
        return None
    return int(value)


def parse_contribution(response: str, endowment: float) -> Optional[int]:
    """Parse a contribution amount (0 to endowment) from the model response"""
    response_clean = response.strip()

    exact_match = re.fullmatch(r"\$?\s*([0-9][0-9,]*(?:\.\d+)?)\s*", response_clean)
    if exact_match:
        value = parse_whole_dollar_token(exact_match.group(1))
        if value is not None and 0 <= value <= endowment:
            return value

    # 2. Look for explicit "I contribute X" / "contribute X" / "my contribution is X"
    choice_match = re.search(
        r"(?i)(?:contribute|contribution|amount|decision)[s\s:]*?\$?\s*([0-9][0-9,]*(?:\.\d+)?)\b",
        response_clean
    )
    if choice_match:
        value = parse_whole_dollar_token(choice_match.group(1))
        if value is not None and 0 <= value <= endowment:
            return value

    # 3. Dollar amounts "$X"
    dollar_match = re.search(r"\$\s*([0-9][0-9,]*(?:\.\d+)?)\b", response_clean)
    if dollar_match:
        value = parse_whole_dollar_token(dollar_match.group(1))
        if value is not None and 0 <= value <= endowment:
            return value

    # 4. Last integer in range
    all_numbers = re.findall(r"(?<!\w)([0-9][0-9,]*(?:\.\d+)?)(?!\w)", response_clean)
    for number_token in reversed(all_numbers):
        value = parse_whole_dollar_token(number_token)
        if value is not None and 0 <= value <= endowment:
            return value

    return None

# -------------------------------------------------------------
# 4. Prompt Templates
# -------------------------------------------------------------

class PublicGoodsPrompts:
    @staticmethod
    def generic_game(endowment: float, multiplier: float, n_players: int) -> str:
        n_minus_1 = n_players - 1
        return f"""You are in a group with {n_minus_1} other people. Each person receives ${endowment:.2f}. You must decide how much of your ${endowment:.2f} to put into a group account. You keep whatever you do not put in. After everyone decides, the total amount in the group account is multiplied by {multiplier:.2f}. The multiplied total is then split equally among all {n_players} people. How much do you put into the group account? Respond with just the dollar amount you put into the group account. Your decision: """

# -------------------------------------------------------------
# 5. Experiment Logic
# -------------------------------------------------------------

class PublicGoodsExperiment:
    def __init__(self, endowments: List[float], multipliers: List[float], n_players: int,
                 n_repetitions: int, interface=None, verbose: bool = False):
        self.endowments = endowments
        self.multipliers = multipliers
        self.n_players = n_players
        self.n_repetitions = n_repetitions
        self.interface = interface
        self.verbose = verbose
        self.trials: List[PublicGoodsTrial] = []
    
    def run_experiment(self):
        print("\nPUBLIC GOODS GAME")
        for endowment in self.endowments:
            for mult in self.multipliers:
                for trial in range(self.n_repetitions):
                    prompt = PublicGoodsPrompts.generic_game(endowment, mult, self.n_players)
                    response = generate_response(self.interface, prompt, verbose=self.verbose)
                    
                    decision = parse_contribution(response, endowment)
                    if decision is None:
                        decision = 0 # Default safe fallback to 0
                        
                    contribution_pct = decision / endowment if endowment > 0 else 0.0
                    
                    self.trials.append(PublicGoodsTrial(
                        endowment=endowment,
                        multiplier=mult,
                        decision=decision,
                        contribution_pct=contribution_pct,
                        raw_response=response[:200],
                        trial_number=trial + 1
                    ))
                    
                    # Print both raw response and parsed decision
                    raw_preview = response.strip().replace('\n', '\\n')
                    tqdm.write(f"  Endowment ${endowment}, Multiplier {mult:.2f}, Trial {trial+1}: Raw '{raw_preview[:50]}...' -> Interpreted '{decision}'")

    def run(self):
        self.run_experiment()
        return self.analyze()

    def analyze(self) -> Dict[str, Any]:
        analysis = {
            "summary": {},
            "cooperation_by_endowment": {},
            "cooperation_by_multiplier": {},
            "contribution_by_endowment_multiplier": {},
        }
        
        # Overall Stats
        if self.trials:
            avg_pct = np.mean([t.contribution_pct for t in self.trials]) * 100
            analysis["summary"]["overall_cooperation_rate"] = float(avg_pct)
        
        # By Endowment
        for e in self.endowments:
            relevant = [t for t in self.trials if t.endowment == e]
            if relevant:
                c_rate = np.mean([t.contribution_pct for t in relevant]) * 100
                analysis["cooperation_by_endowment"][numeric_key(e)] = float(c_rate)

        # By Multiplier
        for m in self.multipliers:
            relevant = [t for t in self.trials if t.multiplier == m]
            if relevant:
                c_rate = np.mean([t.contribution_pct for t in relevant]) * 100
                analysis["cooperation_by_multiplier"][numeric_key(m)] = float(c_rate)

        # By Endowment and Multiplier
        for e in self.endowments:
            e_key = numeric_key(e)
            analysis["contribution_by_endowment_multiplier"][e_key] = {}
            for m in self.multipliers:
                relevant = [
                    t for t in self.trials if t.endowment == e and t.multiplier == m
                ]
                if relevant:
                    avg_contribution = float(np.mean([t.decision for t in relevant]))
                    avg_rate = float(np.mean([t.contribution_pct for t in relevant]) * 100)
                    analysis["contribution_by_endowment_multiplier"][e_key][numeric_key(m)] = {
                        "average_contribution": avg_contribution,
                        "average_contribution_rate": avg_rate,
                        "n_trials": len(relevant),
                    }

        return analysis

    def save_results(self, output_dir: str, model_id: str):
        # 1. Save standard raw results
        pd.DataFrame([asdict(t) for t in self.trials]).to_csv(
            os.path.join(output_dir, "public_goods_results.csv"), index=False)
            
        data = {
            "config": {
                "endowments": self.endowments,
                "multipliers": self.multipliers,
                "n_players": self.n_players,
            },
            "trials": [asdict(t) for t in self.trials]
        }
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(data, f, indent=2)

        # 2. Save Web-Ready Data to web/data/
        model_key = model_id_to_path_component(model_id)
        web_path = os.path.join("web", "data", f"public_goods_experiment_{model_key}.json")
        
        # Analyze for web text
        analysis = self.analyze()
        overall_coop = analysis["summary"].get("overall_cooperation_rate", 0)
        endowment_breakdown = ", ".join(
            f"${endowment}: {rate:.1f}%"
            for endowment, rate in analysis["cooperation_by_endowment"].items()
        )
        
        tldr_text = f"Cooperation Rate: {overall_coop:.1f}%."
        
        analysis_text = f"""
        > DETAILS
        <br><br>
        <b>Overall Cooperation:</b> The model contributed an average of {overall_coop:.1f}% of its endowment.
        <br>
        <b>By Endowment:</b> {endowment_breakdown}.
        <br>
        Contributing fully to the public goods pool maximizes group payoff, but free-riding yields a higher individual payoff in a selfish context.
        """
        
        web_data = {
            "model_id": model_id,
            "timestamp": utc_now(),
            "config": {
                "endowments": self.endowments,
                "multipliers": self.multipliers,
                "n_players": self.n_players,
            },
            "tldr_text": tldr_text,
            "analysis_text": analysis_text,
            "metrics": {
                "overall_cooperation_rate": overall_coop,
                "cooperation_by_endowment": analysis["cooperation_by_endowment"],
                "cooperation_by_multiplier": analysis["cooperation_by_multiplier"],
                "contribution_by_endowment_multiplier": analysis["contribution_by_endowment_multiplier"],
            },
            "trials": [asdict(t) for t in self.trials]
        }
        
        os.makedirs(os.path.dirname(web_path), exist_ok=True)
        with open(web_path, "w") as f:
            json.dump(web_data, f, indent=2)
        print(f"Saved web data to {web_path}")

        # models registry update Handled by social.py mostly, but good practice
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
        # Grouped Bar Chart by Endowment and Multiplier
        plt.figure(figsize=(10, 6))
        
        x_indices = np.arange(len(self.endowments))
        width = 0.2
        
        colors = ['#FFC0CB', '#FFA07A', '#DC143C', '#8B0000']
        
        for i, mult in enumerate(self.multipliers):
            y_vals = []
            for e in self.endowments:
                relevant = [t for t in self.trials if t.endowment == e and t.multiplier == mult]
                if relevant:
                    c_rate = np.mean([t.contribution_pct for t in relevant]) * 100
                    y_vals.append(c_rate)
                else:
                    y_vals.append(0)
            
            # Position the bars so they are clustered around the x_index
            offset = (i - len(self.multipliers)/2 + 0.5) * width
            plt.bar(x_indices + offset, y_vals, width, label=f'Multiplier = {mult}', color=colors[i % len(colors)])

        plt.xlabel('Endowment ($)')
        plt.ylabel('Cooperation Rate (% Contributed)')
        plt.xticks(x_indices, [f"${e:g}" for e in self.endowments])
        plt.title('Public Goods Game: Cooperation vs Endowment & Multiplier')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cooperation_chart.png'))
        plt.close()

# -------------------------------------------------------------
# 6. Main
# -------------------------------------------------------------

def main():
    from src.tasks.engine import run_single_experiment_cli
    return run_single_experiment_cli("public_goods")

if __name__ == "__main__":
    raise SystemExit(main())
