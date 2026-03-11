"""
Beauty Contest Game Experiment (p-Beauty)

This script measures an LLM's strategic depth by simulating a classic 
beauty contest game where the winning number is 2/3 of the average.

Prompt:
You are playing a game with a group of 99 other players. Each player in the group picks a whole number from 0 to 100.
The winner is the player whose number is closest to two thirds of the average of all players' numbers. 
The winner receives ${PRIZE:.2f}.
What number do you pick?
Respond with just your chosen number (a whole number from 0 to 100).
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

PRIZES = [10, 100, 1000]

@dataclass
class BeautyContestTrial:
    prize: int
    decision: int  # The chosen number 0-100
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

def parse_number(response: str) -> Optional[int]:
    """Parse an integer from 0 to 100 from the model response"""
    response_clean = response.strip()
    
    # 1. Exact match
    if response_clean.isdigit() and 0 <= int(response_clean) <= 100:
        return int(response_clean)
        
    # 2. Look for explicit "Your choice: X" or "My number is X"
    choice_match = re.search(r'(?i)(?:choice|pick|number)[\s:]*([0-9]{1,3})\b', response_clean)
    if choice_match:
        val = int(choice_match.group(1))
        if 0 <= val <= 100:
            return val
            
    # 3. Last number in string
    end_matches = re.findall(r'\b([0-9]{1,3})\b', response_clean)
    if end_matches:
        for match in reversed(end_matches):
            val = int(match)
            if 0 <= val <= 100:
                return val

    # 4. Fallback search
    all_numbers = re.findall(r'\b(\d+)\b', response_clean)
    for num_str in all_numbers:
        val = int(num_str)
        if 0 <= val <= 100:
            return val
            
    return None

# -------------------------------------------------------------
# 4. Prompt Templates
# -------------------------------------------------------------

class BeautyContestPrompts:
    @staticmethod
    def generic_game(prize: int) -> str:
        return f"""You are playing a game with a group of 99 other players. Each player in the group picks a whole number from 0 to 100.
The winner is the player whose number is closest to two thirds of the average of all players' numbers. 
The winner receives ${prize:.2f}.
What number do you pick?
Respond with just your chosen number (a whole number from 0 to 100)."""

# -------------------------------------------------------------
# 5. Experiment Logic
# -------------------------------------------------------------

class BeautyContestExperiment:
    def __init__(self, prizes: List[int], n_repetitions: int):
        self.prizes = prizes
        self.n_repetitions = n_repetitions
        self.trials = []
    
    def run_experiment(self):
        print("\nBEAUTY CONTEST GAME (p-Beauty)")
        for prize in self.prizes:
            for trial in range(self.n_repetitions):
                prompt = BeautyContestPrompts.generic_game(prize)
                response = generate_response(prompt)
                
                decision = parse_number(response)
                if decision is None:
                    decision = 50 # Default safe fallback to average baseline if total failure
                
                self.trials.append(BeautyContestTrial(
                    prize=prize,
                    decision=decision,
                    raw_response=response[:200],
                    trial_number=trial + 1
                ))
                
                # Print output
                raw_preview = response.strip().replace('\n', '\\n')
                tqdm.write(f"  Prize ${prize}, Trial {trial+1}: Raw '{raw_preview[:50]}...' -> Parsed: {decision}")

    def run(self):
        self.run_experiment()
        return self.analyze()

    def analyze(self) -> Dict[str, Any]:
        analysis = {"summary": {}, "guesses_by_prize": {}}
        
        all_guesses = [t.decision for t in self.trials]
        if all_guesses:
            analysis["summary"]["overall_average_guess"] = float(np.mean(all_guesses))
            analysis["summary"]["overall_median_guess"] = float(np.median(all_guesses))
            
        for p in self.prizes:
            relevant = [t.decision for t in self.trials if t.prize == p]
            if relevant:
                analysis["guesses_by_prize"][p] = {
                    "mean": float(np.mean(relevant)),
                    "median": float(np.median(relevant)),
                    "min": min(relevant),
                    "max": max(relevant)
                }

        return analysis

    def save_results(self, output_dir: str, model_id: str):
        # 1. Save standard raw results
        pd.DataFrame([vars(t) for t in self.trials]).to_csv(
            os.path.join(output_dir, "beauty_contest_results.csv"), index=False)
            
        data = {
            "trials": [vars(t) for t in self.trials]
        }
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(data, f, indent=2)

        # 2. Save Web-Ready Data to web/data/
        model_safe = model_id.replace("/", "_").replace(":", "_")
        web_path = os.path.join("web", "data", f"beauty_contest_experiment_{model_safe}.json")
        
        # Analyze for web text
        analysis = self.analyze()
        overall_avg = analysis["summary"].get("overall_average_guess", 0)
        
        tldr_text = f"Average Guess: {overall_avg:.1f}."
        
        # Determine depth based on guess level. 50 = level 0, 33 = level 1, 22 = level 2, 0 = NE
        if overall_avg <= 5: depth_str = "Infinite/Nash Equilibrium (0)"
        elif overall_avg <= 22: depth_str = "Level 2 or 3 Thinker (14-22)"
        elif overall_avg <= 33: depth_str = "Level 1 Thinker (33)"
        else: depth_str = "Level 0 Thinker (>33)"
        
        analysis_text = f"""
        > DETAILS
        <br><br>
        <b>Overall Strategy:</b> The model chose an average number of {overall_avg:.1f}.
        <br>
        <b>Strategic Depth:</b> {depth_str}. Random selection averages to 50. If everyone knows this, they pick 33. If they know others will pick 33, they pick 22. The Nash Equilibrium is 0.
        """
        
        web_data = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "tldr_text": tldr_text,
            "analysis_text": analysis_text,
            "metrics": {
                "overall_average": overall_avg
            },
            "trials": [vars(t) for t in self.trials]
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
        # Scatter Plot of 0-100 Guesses
        plt.figure(figsize=(10, 6))
        
        x_indices = np.arange(len(self.prizes))
        
        # Add a baseline indicating levels of reasoning
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.3, label='Level 0 (50)')
        plt.axhline(y=33.3, color='g', linestyle='--', alpha=0.3, label='Level 1 (33)')
        plt.axhline(y=22.2, color='b', linestyle='--', alpha=0.3, label='Level 2 (22)')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Nash Eq (0)')

        for i, prize in enumerate(self.prizes):
            relevant_trials = [t.decision for t in self.trials if t.prize == prize]
            
            # Add some jitter to x for scatter plot visibility
            jitter = np.random.normal(0, 0.05, len(relevant_trials))
            
            plt.scatter([i] * len(relevant_trials) + jitter, relevant_trials, 
                        alpha=0.6, edgecolors='k', color='coral')

        plt.xlabel('Prize Pool ($)')
        plt.ylabel('Chosen Number (0-100)')
        plt.xticks(x_indices, [f"${p}" for p in self.prizes])
        plt.title('Beauty Contest Guesses vs Prize Pool')
        plt.ylim(-5, 105)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'beauty_contest_scatter.png'))
        plt.close()

# -------------------------------------------------------------
# 6. Main
# -------------------------------------------------------------

def main():
    global llm, PRINT_INTERACTIONS
    
    parser = argparse.ArgumentParser(description="Beauty Contest Experiment")
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
    output_dir = os.path.join("data", "results", "beauty_contest", args.model.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Experiment
    exp = BeautyContestExperiment(
        prizes=PRIZES,
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
