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
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

# -------------------------------------------------------------
# 3. Helper Functions
# -------------------------------------------------------------

def generate_response(prompt: str, temperature: float = 0.5) -> str:
    """Generate response using the global LLM interface"""
    response, _ = llm.generate_response(
        prompt=prompt,
        max_new_tokens=8192,
        temperature=temperature,
        verbose=PRINT_INTERACTIONS
    )
    return response

def parse_a_b(response: str) -> Optional[str]:
    """Parse A or B from model response"""
    response_clean = response.strip()
    response_upper = response_clean.upper()
    
    # 1. Exact match or starts with
    if response_upper.startswith("A"): return "A"
    if response_upper.startswith("B"): return "B"
    
    # 2. Look for a labeled answer like "Your decision: A"
    choice_match = re.search(r'(?i)(?:choice|decision|answer):\s*([AB])\b', response_clean)
    if choice_match:
        return choice_match.group(1).upper()
        
    # 3. Look for standalone A or B at the very end of the string
    # e.g. "I have thought about this carefully. B"
    end_match = re.search(r'\b([AB])\W*$', response_upper)
    if end_match:
        return end_match.group(1)
        
    # 4. General word boundary search (risky if text contains "a" as an article, 
    # but we look for uppercase A/B in the original, or just fallback to upper)
    patterns = [
        (r'\bA\b', "A"),
        (r'\bB\b', "B"),
    ]
    
    for pattern, result in patterns:
        if re.search(pattern, response_upper):
            # If both A and B might be in the text, this just returns the first one it finds.
            # To be safer, we could count or look at the last occurrence.
            return result
            
    return None

# -------------------------------------------------------------
# 4. Prompt Templates
# -------------------------------------------------------------

class StagHuntPrompts:
    @staticmethod
    def template() -> str:
        return """You and another person each make a single choice at the same time, without knowing what the other will choose. After both choices are made, the game is over.
If you choose A: you earn ${SAFE_PAYOFF:.2f} no matter what the other person chooses.
If you choose B: you earn ${COORDINATION_PAYOFF:.2f} if they also choose B, or ${MISCOORDINATION_PAYOFF:.2f} if they choose A.
What do you choose?
Respond with only "A" or "B".
Your decision:"""

    @staticmethod
    def default() -> str:
        return StagHuntPrompts.generic_stag_hunt(
            payoff=DEFAULT_BASE_PAYOFF,
            x_multiplier=DEFAULT_X_MULTIPLIER,
            miscoordination_payoff=DEFAULT_MISCOORDINATION_PAYOFF
        )

    @staticmethod
    def generic_stag_hunt(payoff: int, x_multiplier: float,
                          miscoordination_payoff: float = DEFAULT_MISCOORDINATION_PAYOFF) -> str:
        safe_payoff = payoff * x_multiplier
        return f"""You and another person each make a single choice at the same time, without knowing what the other will choose. After both choices are made, the game is over.
If you choose A: you earn ${safe_payoff:.2f} no matter what the other person chooses.
If you choose B: you earn ${payoff:.2f} if they also choose B, or ${miscoordination_payoff:.2f} if they choose A.
What do you choose?
Respond with only "A" or "B".
Your decision:"""

# -------------------------------------------------------------
# 5. Experiment Logic
# -------------------------------------------------------------

class StagHuntExperiment:
    def __init__(self, payoffs: List[int], x_multipliers: List[float], n_repetitions: int):
        self.payoffs = payoffs
        self.x_multipliers = x_multipliers
        self.n_repetitions = n_repetitions
        self.trials = []
    
    def run_experiment(self):
        print("\nSTAG HUNT GAME")
        for payoff in self.payoffs:
            for x_mult in self.x_multipliers:
                for trial in range(self.n_repetitions):
                    prompt = StagHuntPrompts.generic_stag_hunt(payoff, x_mult)
                    response = generate_response(prompt)
                    
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
        model_safe = model_id.replace("/", "_").replace(":", "_")
        web_path = os.path.join("web", "data", f"stag_hunt_experiment_{model_safe}.json")
        
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
    global llm, PRINT_INTERACTIONS
    
    parser = argparse.ArgumentParser(description="Stag Hunt Game Experiment")
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
    output_dir = os.path.join("data", "results", "stag_hunt", args.model.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Experiment
    exp = StagHuntExperiment(
        payoffs=PAYOFFS,
        x_multipliers=X_MULTIPLIERS,
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
