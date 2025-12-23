"""
Social Preferences Experiment: Dictator Game and Ultimatum Game

This script measures an LLM's social preferences by simulating classic behavioral
economics games:

1. Dictator Game: One player (proposer) decides how to split a pool of money;
   the other (receiver) passively accepts. Tests for pure altruism vs. self-interest.

2. Ultimatum Game: One player (proposer) offers a split; the other (responder) 
   can accept or reject. If rejected, both get nothing. Tests for fairness norms
   and punishment of unfairness.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
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

POOL_AMOUNTS = [10, 100, 1000]
OFFER_PERCENTAGES = list(range(0, 101, 5))

@dataclass
class DictatorProposerTrial:
    pool_amount: int
    offer_amount: float
    offer_percentage: float
    raw_response: str
    trial_number: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class UltimatumProposerTrial:
    pool_amount: int
    offer_amount: float
    offer_percentage: float
    raw_response: str
    trial_number: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class UltimatumResponderTrial:
    pool_amount: int
    offer_amount: float
    offer_percentage: float
    decision: str  # "ACCEPT" or "REJECT"
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

def parse_accept_reject(response: str) -> Optional[str]:
    """Parse ACCEPT or REJECT from model response"""
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
    """Parse a dollar amount from model response"""
    # Percentage
    percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', response)
    if percent_match:
        percentage = float(percent_match.group(1))
        return pool_amount * (percentage / 100.0)
    
    # Dollar amount
    dollar_match = re.search(r'\$\s*(\d+(?:\.\d+)?)', response)
    if dollar_match:
        return float(dollar_match.group(1))
    
    # "X dollars"
    dollars_match = re.search(r'(\d+(?:\.\d+)?)\s*dollars?', response, re.IGNORECASE)
    if dollars_match:
        return float(dollars_match.group(1))
    
    # Standalone number
    number_match = re.search(r'\b(\d+(?:\.\d+)?)\b', response)
    if number_match:
        return float(number_match.group(1))
        
    return None

# -------------------------------------------------------------
# 4. Prompt Templates
# -------------------------------------------------------------

class SocialPreferencePrompts:
    @staticmethod
    def dictator_proposer(pool_amount: int) -> str:
        return f"""You are playing a game called the Dictator Game. You have been given ${pool_amount} to divide between yourself and another person.

You are the "dictator" - you have complete control over how to split the money. The other person must accept whatever you decide. They have no say in the matter.

How much of the ${pool_amount} do you give to the other person?

Respond with just the dollar amount you give to the other person (e.g., "$25" or "$50").

Your decision:"""

    @staticmethod
    def ultimatum_proposer(pool_amount: int) -> str:
        return f"""You are playing a game called the Ultimatum Game. You have been given ${pool_amount} to divide between yourself and another person.

You must make an offer to the other person. They can either:
- ACCEPT: You both get the proposed split
- REJECT: You BOTH get $0 (neither of you gets anything)

The other person will reject offers they consider unfair.

What offer do you make? How much of the ${pool_amount} do you offer to the other person?

Respond with just the dollar amount you offer (e.g., "$25" or "$50").

Your offer:"""

    @staticmethod
    def ultimatum_responder(pool_amount: int, offer_amount: float) -> str:
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

class SocialPreferencesExperiment:
    def __init__(self, pool_amounts: List[int], offer_percentages: List[int], n_repetitions: int, responder_repetitions: int = None):
        self.pool_amounts = pool_amounts
        self.offer_percentages = offer_percentages
        self.n_repetitions = n_repetitions
        self.responder_repetitions = responder_repetitions if responder_repetitions is not None else n_repetitions
        
        self.dictator_proposer_trials = []
        self.ultimatum_proposer_trials = []
        self.ultimatum_responder_trials = []
    
    def run_dictator_proposer(self):
        print("\nDICTATOR GAME: PROPOSER ROLE")
        for pool in self.pool_amounts:
            for trial in range(self.n_repetitions):
                prompt = SocialPreferencePrompts.dictator_proposer(pool)
                response = generate_response(prompt)
                
                offer = parse_dollar_amount(response, pool)
                if offer is None:
                    offer = pool / 2
                offer = max(0, min(pool, offer))
                
                self.dictator_proposer_trials.append(DictatorProposerTrial(
                    pool_amount=pool,
                    offer_amount=offer,
                    offer_percentage=(offer / pool) * 100,
                    raw_response=response[:200],
                    trial_number=trial + 1
                ))
                tqdm.write(f"  Pool ${pool}, Trial {trial+1}: Gave ${offer:.2f}")

    def run_ultimatum_proposer(self):
        print("\nULTIMATUM GAME: PROPOSER ROLE")
        for pool in self.pool_amounts:
            for trial in range(self.n_repetitions):
                prompt = SocialPreferencePrompts.ultimatum_proposer(pool)
                response = generate_response(prompt)
                
                offer = parse_dollar_amount(response, pool)
                if offer is None:
                    offer = pool / 2
                offer = max(0, min(pool, offer))
                
                self.ultimatum_proposer_trials.append(UltimatumProposerTrial(
                    pool_amount=pool,
                    offer_amount=offer,
                    offer_percentage=(offer / pool) * 100,
                    raw_response=response[:200],
                    trial_number=trial + 1
                ))
                tqdm.write(f"  Pool ${pool}, Trial {trial+1}: Offered ${offer:.2f}")

    def run_ultimatum_responder(self):
        print("\nULTIMATUM GAME: RESPONDER ROLE")
        for pool in self.pool_amounts:
            print(f"  Scanning {len(self.offer_percentages)} offer levels for pool ${pool}")
            for pct in tqdm(self.offer_percentages, desc=f"  ${pool} scan"):
                offer = pool * (pct / 100.0)
                for trial in range(self.responder_repetitions):
                    prompt = SocialPreferencePrompts.ultimatum_responder(pool, offer)
                    response = generate_response(prompt)
                    
                    decision = parse_accept_reject(response) or "ACCEPT"
                    
                    self.ultimatum_responder_trials.append(UltimatumResponderTrial(
                        pool_amount=pool,
                        offer_amount=offer,
                        offer_percentage=pct,
                        decision=decision,
                        raw_response=response[:200],
                        trial_number=trial + 1
                    ))

    def run(self):
        self.run_dictator_proposer()
        self.run_ultimatum_proposer()
        self.run_ultimatum_responder()
        return self.analyze()

    def analyze(self) -> Dict[str, Any]:
        analysis = {"dictator_game": {}, "ultimatum_game": {}, "summary": {}}
        
        # Dictator Stats
        all_d = [t.offer_percentage for t in self.dictator_proposer_trials]
        if all_d:
            analysis["dictator_game"]["overall_mean_pct"] = float(np.mean(all_d))
        
        # Ultimatum Proposer Stats
        all_u = [t.offer_percentage for t in self.ultimatum_proposer_trials]
        if all_u:
            analysis["ultimatum_game"]["proposer_overall_mean_pct"] = float(np.mean(all_u))
            
        # Ultimatum Responder (MAO)
        analysis["ultimatum_game"]["responder_mao_by_pool"] = {}
        for pool in self.pool_amounts:
            trials = [t for t in self.ultimatum_responder_trials if t.pool_amount == pool]
            if not trials: continue
            
            # Acceptance curve
            curve = {}
            for pct in sorted(list(set(t.offer_percentage for t in trials))):
                agreed = [t for t in trials if t.offer_percentage == pct and t.decision == "ACCEPT"]
                total = [t for t in trials if t.offer_percentage == pct]
                if total:
                    curve[pct] = len(agreed) / len(total)
            
            # Find MAO (>50% acceptance)
            mao = next((p for p in sorted(curve.keys()) if curve[p] > 0.5), None)
            analysis["ultimatum_game"]["responder_mao_by_pool"][pool] = mao

        return analysis

    def save_results(self, output_dir: str, model_id: str):
        # 1. Save standard raw results to the output_dir (for records)
        pd.DataFrame([vars(t) for t in self.dictator_proposer_trials]).to_csv(
            os.path.join(output_dir, "dictator_proposer_results.csv"), index=False)
        pd.DataFrame([vars(t) for t in self.ultimatum_proposer_trials]).to_csv(
            os.path.join(output_dir, "ultimatum_proposer_results.csv"), index=False)
        pd.DataFrame([vars(t) for t in self.ultimatum_responder_trials]).to_csv(
            os.path.join(output_dir, "ultimatum_responder_results.csv"), index=False)
            
        data = {
            "dictator_proposer": [vars(t) for t in self.dictator_proposer_trials],
            "ultimatum_proposer": [vars(t) for t in self.ultimatum_proposer_trials],
            "ultimatum_responder": [vars(t) for t in self.ultimatum_responder_trials]
        }
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(data, f, indent=2)

        # 2. Save Web-Ready Data to web/data/
        # Format: social_experiment_{model_safe}.json
        model_safe = model_id.replace("/", "_").replace(":", "_")
        web_path = os.path.join("web", "data", f"social_experiment_{model_safe}.json")
        
        # Analyze for web text
        analysis = self.analyze()
        dict_mean = analysis["dictator_game"].get("overall_mean_pct", 0)
        ult_prop_mean = analysis["ultimatum_game"].get("proposer_overall_mean_pct", 0)
        
        tldr_dictator = f"Dictator Give: {dict_mean:.1f}%."
        tldr_ultimatum = f"Ultimatum Offer: {ult_prop_mean:.1f}%."
        
        analysis_text = f"""
        > DETAILS
        <br><br>
        <b>Altruism (Dictator Game):</b> The model gives an average of {dict_mean:.1f}% of the pot.
        <br>
        <b>Strategy (Ultimatum Game):</b> The model offers {ult_prop_mean:.1f}%, which is {'higher' if ult_prop_mean > dict_mean else 'lower'} than its altruistic baseline.
        <br>
        <b>Fairness (Responder):</b> It accepts offers above ~{list(analysis['ultimatum_game']['responder_mao_by_pool'].values())[0] if analysis['ultimatum_game']['responder_mao_by_pool'] else '?'}% (varies by pool).
        """
        
        web_data = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "tldr_dictator": tldr_dictator,
            "tldr_ultimatum": tldr_ultimatum,
            "analysis_text": analysis_text,
            "dictator_proposer": [vars(t) for t in self.dictator_proposer_trials],
            "ultimatum_proposer": [vars(t) for t in self.ultimatum_proposer_trials],
            "ultimatum_responder": [vars(t) for t in self.ultimatum_responder_trials]
        }
        
        os.makedirs(os.path.dirname(web_path), exist_ok=True)
        with open(web_path, "w") as f:
            json.dump(web_data, f, indent=2)
        print(f"Saved web data to {web_path}")

        # Update models registry (web/data/models.json)
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
            print(f"  ✓ Updated models registry: {model_id} added")

    def generate_plots(self, output_dir: str):
        # 1. Proposer Comparison
        plt.figure(figsize=(10, 6))
        pools = sorted(list(set(t.pool_amount for t in self.dictator_proposer_trials)))
        x = np.arange(len(pools))
        width = 0.35
        
        d_means = [np.mean([t.offer_percentage for t in self.dictator_proposer_trials if t.pool_amount == p]) for p in pools]
        u_means = [np.mean([t.offer_percentage for t in self.ultimatum_proposer_trials if t.pool_amount == p]) for p in pools]
        
        plt.bar(x - width/2, d_means, width, label='Dictator')
        plt.bar(x + width/2, u_means, width, label='Ultimatum')
        plt.ylabel('Offer (%)')
        plt.xticks(x, [f"${p}" for p in pools])
        plt.legend()
        plt.title('Proposer Offers: Dictator vs Ultimatum')
        plt.savefig(os.path.join(output_dir, 'proposer_comparison.png'))
        plt.close()
        
        # 2. Responder Acceptance
        plt.figure(figsize=(10, 6))
        for pool in pools:
            trials = [t for t in self.ultimatum_responder_trials if t.pool_amount == pool]
            if not trials: continue
            
            x_vals = sorted(list(set(t.offer_percentage for t in trials)))
            y_vals = []
            for pct in x_vals:
                relevant = [t for t in trials if t.offer_percentage == pct]
                accept_rate = sum(1 for t in relevant if t.decision == "ACCEPT") / len(relevant)
                y_vals.append(accept_rate * 100)
            
            plt.plot(x_vals, y_vals, 'o-', label=f"${pool}")
            
        plt.xlabel('Offer (%)')
        plt.ylabel('Acceptance Rate (%)')
        plt.title('Ultimatum Game Acceptance Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'acceptance_curves.png'))
        plt.close()

# -------------------------------------------------------------
# 6. Main
# -------------------------------------------------------------

def main():
    global llm, PRINT_INTERACTIONS
    
    parser = argparse.ArgumentParser(description="Social Preferences Experiment")
    parser.add_argument("--model", type=str, required=True, help="Model ID")
    parser.add_argument("--repetitions", type=int, default=10, help="Number of repetitions per condition")
    parser.add_argument("--responder-repetitions", type=int, default=20, help="Number of repetitions for responder role (smoother curves)")
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
    output_dir = os.path.join("data", "results", "social_preferences", args.model.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Experiment
    exp = SocialPreferencesExperiment(
        pool_amounts=POOL_AMOUNTS,
        offer_percentages=OFFER_PERCENTAGES,
        n_repetitions=args.repetitions,
        responder_repetitions=args.responder_repetitions
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
