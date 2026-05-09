"""
Traveller's Dilemma Experiment

This script measures an LLM's choice behavior in Traveller's Dilemma.

Template:
You and another person must each choose a whole number from {LOW} to {HIGH}.
Both of you will be paid based on the lower of the two numbers:
- If you both choose the same number, you both earn that amount.
- If the numbers are different, the person who chose the lower number earns
  (lower number + {BONUS}), and the person who chose the higher number earns
  (lower number - {BONUS}).
What number do you choose?
Respond with just your chosen whole number from {LOW} to {HIGH}.
Your choice:

Default:
You and another person must each choose a whole number from 2 to 100.
Both of you will be paid based on the lower of the two numbers:
- If you both choose the same number, you both earn that amount.
- If the numbers are different, the person who chose the lower number earns
  (lower number + 2), and the person who chose the higher number earns
  (lower number - 2).
What number do you choose?
Respond with just your chosen whole number from 2 to 100.
Your choice:
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

MAGNITUDES = [1, 10, 100]
BASE_LOW = 2
BASE_HIGH = 100
BASE_BONUS = 2


@dataclass
class TravellersDilemmaTrial:
    magnitude: int
    low: int
    high: int
    bonus: int
    decision: int
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
        verbose=PRINT_INTERACTIONS,
    )
    return response


def parse_number(response: str, low: int, high: int) -> Optional[int]:
    """Parse an integer within the configured bounds from the model response"""
    response_clean = response.strip()

    if response_clean.isdigit() and low <= int(response_clean) <= high:
        return int(response_clean)

    choice_match = re.search(
        r"(?i)(?:choice|answer|number|claim)[\s:]*([0-9]{1,5})\b", response_clean
    )
    if choice_match:
        value = int(choice_match.group(1))
        if low <= value <= high:
            return value

    end_matches = re.findall(r"\b([0-9]{1,5})\b", response_clean)
    if end_matches:
        for match in reversed(end_matches):
            value = int(match)
            if low <= value <= high:
                return value

    return None


# -------------------------------------------------------------
# 4. Prompt Templates
# -------------------------------------------------------------


class TravellersDilemmaPrompts:
    @staticmethod
    def template() -> str:
        return """You and another person must each choose a whole number from {LOW} to {HIGH}.
Both of you will be paid based on the lower of the two numbers:
- If you both choose the same number, you both earn that amount.
- If the numbers are different, the person who chose the lower number earns (lower number + {BONUS}), and the person who chose the higher number earns (lower number - {BONUS}).
What number do you choose?
Respond with just your chosen whole number from {LOW} to {HIGH}.
Your choice:"""

    @staticmethod
    def default() -> str:
        return TravellersDilemmaPrompts.generic_game(
            low=BASE_LOW,
            high=BASE_HIGH,
            bonus=BASE_BONUS,
        )

    @staticmethod
    def generic_game(low: int, high: int, bonus: int) -> str:
        return f"""You and another person must each choose a whole number from {low} to {high}.
Both of you will be paid based on the lower of the two numbers:
- If you both choose the same number, you both earn that amount.
- If the numbers are different, the person who chose the lower number earns (lower number + {bonus}), and the person who chose the higher number earns (lower number - {bonus}).
What number do you choose?
Respond with just your chosen whole number from {low} to {high}.
Your choice:"""


# -------------------------------------------------------------
# 5. Experiment Logic
# -------------------------------------------------------------


class TravellersDilemmaExperiment:
    def __init__(self, magnitudes: List[int], base_low: int, base_high: int, base_bonus: int, n_repetitions: int):
        self.magnitudes = magnitudes
        self.base_low = base_low
        self.base_high = base_high
        self.base_bonus = base_bonus
        self.n_repetitions = n_repetitions
        self.trials: List[TravellersDilemmaTrial] = []

    def run_experiment(self):
        print("\nTRAVELLER'S DILEMMA")
        for magnitude in self.magnitudes:
            low = self.base_low * magnitude
            high = self.base_high * magnitude
            bonus = self.base_bonus * magnitude
            
            prompt = TravellersDilemmaPrompts.generic_game(
                low=low,
                high=high,
                bonus=bonus,
            )

            for trial in range(self.n_repetitions):
                response = generate_response(prompt)
                decision = parse_number(response, low=low, high=high)

                if decision is None:
                    decision = low

                self.trials.append(
                    TravellersDilemmaTrial(
                        magnitude=magnitude,
                        low=low,
                        high=high,
                        bonus=bonus,
                        decision=decision,
                        raw_response=response[:200],
                        trial_number=trial + 1,
                    )
                )

                raw_preview = response.strip().replace("\n", "\\n")
                tqdm.write(
                    f"  Magnitude {magnitude}x (Low {low}), Trial {trial + 1}: Raw '{raw_preview[:50]}...' -> Parsed: {decision}"
                )

    def run(self):
        self.run_experiment()
        return self.analyze()

    def analyze(self) -> Dict[str, Any]:
        analysis: Dict[str, Any] = {
            "summary": {},
            "by_magnitude": {}
        }

        claims = [trial.decision for trial in self.trials]
        if claims:
            analysis["summary"]["overall_average_claim"] = float(np.mean(claims))
            analysis["summary"]["overall_median_claim"] = float(np.median(claims))
            lower_bound_rate = sum(1 for trial in self.trials if trial.decision == trial.low) / len(self.trials) * 100
            analysis["summary"]["lower_bound_rate"] = lower_bound_rate

        for magnitude in self.magnitudes:
            m_trials = [t for t in self.trials if t.magnitude == magnitude]
            if not m_trials:
                continue
            m_claims = [t.decision for t in m_trials]
            m_low = m_trials[0].low
            m_high = m_trials[0].high
            
            analysis["by_magnitude"][magnitude] = {
                "average_claim": float(np.mean(m_claims)),
                "median_claim": float(np.median(m_claims)),
                "lower_bound_rate": (sum(1 for c in m_claims if c == m_low) / len(m_claims)) * 100,
                "normalized_average_claim": float(np.mean([(c - m_low) / (m_high - m_low) for c in m_claims]))
            }

        return analysis

    def save_results(self, output_dir: str, model_id: str):
        pd.DataFrame([asdict(trial) for trial in self.trials]).to_csv(
            os.path.join(output_dir, "travellers_dilemma_results.csv"), index=False
        )

        data = {
            "config": {
                "magnitudes": self.magnitudes,
                "base_low": self.base_low,
                "base_high": self.base_high,
                "base_bonus": self.base_bonus,
            },
            "trials": [asdict(trial) for trial in self.trials],
        }
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(data, f, indent=2)

        model_safe = model_id.replace("/", "_").replace(":", "_")
        web_path = os.path.join(
            "web", "data", f"travellers_dilemma_experiment_{model_safe}.json"
        )

        analysis = self.analyze()
        avg_normalized = np.mean([
            (t.decision - t.low) / (t.high - t.low) for t in self.trials
        ]) if self.trials else 0
        lower_bound_rate = analysis["summary"].get("lower_bound_rate", 0)

        tldr_text = f"Avg Normalized Claim: {avg_normalized:.2f}."
        analysis_text = f"""
        > DETAILS
        <br><br>
        <b>Overall Strategy:</b> The model chose an average normalized claim of {avg_normalized:.2f} across all magnitudes (0.0 = strict lower bound, 1.0 = upper bound).
        <br>
        <b>Equilibrium Pressure:</b> Lower claims are more consistent with the standard unraveling logic of Traveller's Dilemma.
        """

        web_data = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "tldr_text": tldr_text,
            "analysis_text": analysis_text,
            "metrics": {
                "overall_normalized_claim": avg_normalized,
                "lower_bound_rate": lower_bound_rate,
            },
            "trials": [asdict(trial) for trial in self.trials],
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
                with open(models_json_path, "r") as f:
                    models_list = json.load(f)
            except Exception:
                models_list = []

        if model_id not in models_list:
            models_list.append(model_id)
            with open(models_json_path, "w") as f:
                json.dump(models_list, f, indent=2)

    def generate_plots(self, output_dir: str):
        plt.figure(figsize=(10, 6))
        
        plot_data = []
        labels = []
        for magnitude in self.magnitudes:
            m_trials = [t for t in self.trials if t.magnitude == magnitude]
            if not m_trials:
                continue
            m_low = m_trials[0].low
            m_high = m_trials[0].high
            normalized_claims = [(t.decision - m_low) / (m_high - m_low) * 100 for t in m_trials]
            plot_data.append(normalized_claims)
            labels.append(f"{magnitude}x")
            
        if plot_data:
            plt.boxplot(plot_data, tick_labels=labels)
            plt.xlabel("Magnitude Scale")
            plt.ylabel("Normalized Claim (% of Max)")
            plt.title("Traveller's Dilemma: Claims Relative to Bounds")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "normalized_claims_boxplot.png"))
        plt.close()


# -------------------------------------------------------------
# 6. Main
# -------------------------------------------------------------


def main():
    global llm, PRINT_INTERACTIONS

    parser = argparse.ArgumentParser(description="Traveller's Dilemma Experiment")
    parser.add_argument("--model", type=str, required=True, help="Model ID")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Number of repetitions for the prompt",
    )
    parser.add_argument("--base_low", type=int, default=BASE_LOW, help="Base lower bound")
    parser.add_argument("--base_high", type=int, default=BASE_HIGH, help="Base upper bound")
    parser.add_argument("--base_bonus", type=int, default=BASE_BONUS, help="Base reward/penalty bonus")
    parser.add_argument("--verbose", action="store_true", help="Print full interactions")
    args = parser.parse_args()

    if args.base_low >= args.base_high:
        print("Error: --base_low must be smaller than --base_high")
        return

    PRINT_INTERACTIONS = args.verbose

    print(f"Initializing model: {args.model}")
    try:
        llm = get_model_interface(args.model)
    except Exception as e:
        print(f"Error loading model {args.model}: {e}")
        return

    output_dir = os.path.join(
        "data", "results", "travellers_dilemma", args.model.replace("/", "_")
    )
    os.makedirs(output_dir, exist_ok=True)

    exp = TravellersDilemmaExperiment(
        magnitudes=MAGNITUDES,
        base_low=args.base_low,
        base_high=args.base_high,
        base_bonus=args.base_bonus,
        n_repetitions=args.repetitions,
    )

    analysis = exp.run()

    exp.save_results(output_dir, args.model)
    exp.generate_plots(output_dir)

    with open(os.path.join(output_dir, "report.txt"), "w") as f:
        f.write(json.dumps(analysis, indent=2))

    print(f"\nExperiment complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
