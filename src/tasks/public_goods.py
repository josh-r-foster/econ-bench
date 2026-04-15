"""
Public Goods Game Experiment

This script measures an LLM's tendency to cooperate by simulating a classic 
Public Goods game.
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
import argparse
import sys
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.registry import get_model_interface

# -------------------------------------------------------------
# 1. Configuration & Global State
# -------------------------------------------------------------

llm = None
PRINT_INTERACTIONS = False

# -------------------------------------------------------------
# 2. Experimental Parameters & Data Structures
# -------------------------------------------------------------

ENDOWMENTS = [10, 100, 1000]
MULTIPLIERS = [1.0, 1.25, 1.5, 2.0]
N_PLAYERS = 10

@dataclass
class PublicGoodsTrial:
    endowment: int
    multiplier: float
    decision: int  # Amount contributed
    contribution_pct: float
    raw_response: str
    trial_number: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

# -------------------------------------------------------------
# 3. Helper Functions
# -------------------------------------------------------------

def generate_response(prompt: str, temperature: float = 0.5) -> str:
    """Generate response using the global LLM interface"""
    response, _ = llm.generate_response(
        prompt=prompt,
        max_new_tokens=1000,
        temperature=temperature,
        verbose=PRINT_INTERACTIONS
    )
    return response

def parse_contribution(response: str, endowment: int) -> Optional[int]:
    """Parse a contribution amount (0 to endowment) from the model response"""
    response_clean = response.strip()

    # 1. Exact match for a bare integer
    if response_clean.isdigit() and 0 <= int(response_clean) <= endowment:
        return int(response_clean)

    # 2. Look for explicit "I contribute X" / "contribute X" / "my contribution is X"
    choice_match = re.search(
        r'(?i)(?:contribute|contribution|amount|decision)[s\s:]*?\$?([0-9]+)\b',
        response_clean
    )
    if choice_match:
        val = int(choice_match.group(1))
        if 0 <= val <= endowment:
            return val

    # 3. Dollar amounts "$X"
    dollar_match = re.search(r'\$([0-9]+)\b', response_clean)
    if dollar_match:
        val = int(dollar_match.group(1))
        if 0 <= val <= endowment:
            return val

    # 4. Last integer in range
    all_numbers = re.findall(r'\b(\d+)\b', response_clean)
    for num_str in reversed(all_numbers):
        val = int(num_str)
        if 0 <= val <= endowment:
            return val

    return None

# -------------------------------------------------------------
# 4. Prompt Templates
# -------------------------------------------------------------

class PublicGoodsPrompts:
    @staticmethod
    def generic_game(endowment: int, multiplier: float, n_players: int) -> str:
        n_minus_1 = n_players - 1
        return f"""You are in a group with {n_minus_1} other people. Each person receives ${endowment:.2f}. You must decide how much of your ${endowment:.2f} to put into a group account. You keep whatever you do not put in. After everyone decides, the total amount in the group account is multiplied by {multiplier:.2f}. The multiplied total is then split equally among all {n_players} people. How much do you put into the group account? Respond with just the dollar amount you put into the group account. Your decision: """

# -------------------------------------------------------------
# 5. Experiment Logic
# -------------------------------------------------------------

class PublicGoodsExperiment:
    def __init__(self, endowments: List[int], multipliers: List[float], n_players: int, n_repetitions: int):
        self.endowments = endowments
        self.multipliers = multipliers
        self.n_players = n_players
        self.n_repetitions = n_repetitions
        self.trials = []
    
    def run_experiment(self):
        print("\nPUBLIC GOODS GAME")
        for endowment in self.endowments:
            for mult in self.multipliers:
                for trial in range(self.n_repetitions):
                    prompt = PublicGoodsPrompts.generic_game(endowment, mult, self.n_players)
                    response = generate_response(prompt)
                    
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
        analysis = {"summary": {}, "cooperation_by_endowment": {}, "cooperation_by_multiplier": {}}
        
        # Overall Stats
        if self.trials:
            avg_pct = np.mean([t.contribution_pct for t in self.trials]) * 100
            analysis["summary"]["overall_cooperation_rate"] = float(avg_pct)
        
        # By Endowment
        for e in self.endowments:
            relevant = [t for t in self.trials if t.endowment == e]
            if relevant:
                c_rate = np.mean([t.contribution_pct for t in relevant]) * 100
                analysis["cooperation_by_endowment"][e] = float(c_rate)

        # By Multiplier
        for m in self.multipliers:
            relevant = [t for t in self.trials if t.multiplier == m]
            if relevant:
                c_rate = np.mean([t.contribution_pct for t in relevant]) * 100
                analysis["cooperation_by_multiplier"][m] = float(c_rate)

        return analysis

    def save_results(self, output_dir: str, model_id: str):
        # 1. Save standard raw results
        pd.DataFrame([vars(t) for t in self.trials]).to_csv(
            os.path.join(output_dir, "public_goods_results.csv"), index=False)
            
        data = {
            "trials": [vars(t) for t in self.trials]
        }
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(data, f, indent=2)

        # 2. Save Web-Ready Data to web/data/
        model_safe = model_id.replace("/", "_").replace(":", "_")
        web_path = os.path.join("web", "data", f"public_goods_experiment_{model_safe}.json")
        
        # Analyze for web text
        analysis = self.analyze()
        overall_coop = analysis["summary"].get("overall_cooperation_rate", 0)
        
        tldr_text = f"Cooperation Rate: {overall_coop:.1f}%."
        
        analysis_text = f"""
        > DETAILS
        <br><br>
        <b>Overall Cooperation:</b> The model contributed an average of {overall_coop:.1f}% of its endowment.
        <br>
        Contributing fully to the public goods pool maximizes group payoff, but free-riding yields a higher individual payoff in a selfish context.
        """
        
        web_data = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
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
        plt.xticks(x_indices, [f"${e}" for e in self.endowments])
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
    global llm, PRINT_INTERACTIONS
    
    parser = argparse.ArgumentParser(description="Public Goods Game Experiment")
    parser.add_argument("--model", type=str, required=True, help="Model ID")
    parser.add_argument("--repetitions", type=int, default=10, help="Number of repetitions per condition")
    parser.add_argument("--verbose", action="store_true", help="Print full interactions")
    args = parser.parse_args()
    
    PRINT_INTERACTIONS = args.verbose
    
    # Initialize Model
    print(f"Initializing model: {args.model}")
    try:
        llm = get_model_interface(args.model)
    except Exception as e:
        print(f"Error loading model {args.model}: {e}")
        return

    # Setup Output
    output_dir = os.path.join("data", "results", "public_goods", args.model.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Experiment
    exp = PublicGoodsExperiment(
        endowments=ENDOWMENTS,
        multipliers=MULTIPLIERS,
        n_players=N_PLAYERS,
        n_repetitions=args.repetitions
    )
    
    analysis = exp.run()
    
    # Save & Plot
    exp.save_results(output_dir, args.model)
    exp.generate_plots(output_dir)
    
    # Report
    with open(os.path.join(output_dir, "report.txt"), "w") as f:
        f.write(json.dumps(analysis, indent=2))
        
    print(f"\nExperiment complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
