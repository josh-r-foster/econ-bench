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

MONETARY_LEVELS = [10.0, 100.0, 1000.0]
MAGNITUDES = MONETARY_LEVELS
BASE_LOW = 2
BASE_HIGH = 100
BASE_BONUS = 2
WEB_LOW = 2
WEB_HIGH = 100


@dataclass
class TravellersDilemmaTrial:
    magnitude: float
    monetary_level: float
    low: int
    high: int
    bonus: int
    decision: int
    relative_claim: float
    claim_100_scale: float
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


def parse_whole_number_token(token: str) -> Optional[int]:
    token = token.replace(",", "").strip()
    try:
        value = float(token)
    except ValueError:
        return None

    if not value.is_integer():
        return None
    return int(value)


def parse_number(response: str, low: int, high: int) -> Optional[int]:
    """Parse an integer within the configured bounds from the model response"""
    response_clean = response.strip()

    exact_match = re.fullmatch(r"\$?\s*([0-9][0-9,]*(?:\.\d+)?)\s*", response_clean)
    if exact_match:
        value = parse_whole_number_token(exact_match.group(1))
        if value is not None and low <= value <= high:
            return value

    choice_match = re.search(
        r"(?i)(?:choice|answer|number|claim)[\s:]*\$?\s*([0-9][0-9,]*(?:\.\d+)?)\b",
        response_clean,
    )
    if choice_match:
        value = parse_whole_number_token(choice_match.group(1))
        if value is not None and low <= value <= high:
            return value

    end_matches = re.findall(r"(?<!\w)([0-9][0-9,]*(?:\.\d+)?)(?!\w)", response_clean)
    if end_matches:
        for match in reversed(end_matches):
            value = parse_whole_number_token(match)
            if value is not None and low <= value <= high:
                return value

    return None


def monetary_bounds_for_level(
    monetary_level: float,
    base_low: int,
    base_high: int,
    base_bonus: int,
) -> Tuple[int, int, int]:
    if not float(monetary_level).is_integer():
        raise ValueError(f"Monetary level ${monetary_level:g} must be a whole dollar amount")

    high = int(monetary_level)
    scale = high / base_high
    low = max(base_low, int(round(base_low * scale)))
    bonus = max(base_bonus, int(round(base_bonus * scale)))

    if low >= high:
        raise ValueError(f"Monetary level ${monetary_level:g} must exceed lower bound {low}")

    return low, high, bonus


def claim_on_100_scale(relative_claim: float) -> float:
    return WEB_LOW + relative_claim * (WEB_HIGH - WEB_LOW)


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
    def __init__(self, magnitudes: List[float], base_low: int, base_high: int, base_bonus: int, n_repetitions: int):
        self.magnitudes = magnitudes
        self.monetary_levels = magnitudes
        self.base_low = base_low
        self.base_high = base_high
        self.base_bonus = base_bonus
        self.n_repetitions = n_repetitions
        self.trials: List[TravellersDilemmaTrial] = []

    def run_experiment(self):
        print("\nTRAVELLER'S DILEMMA")
        for monetary_level in self.monetary_levels:
            low, high, bonus = monetary_bounds_for_level(
                monetary_level=monetary_level,
                base_low=self.base_low,
                base_high=self.base_high,
                base_bonus=self.base_bonus,
            )
            
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

                relative_claim = (decision - low) / (high - low) if high > low else 0.0
                claim_100_scale = claim_on_100_scale(relative_claim)

                self.trials.append(
                    TravellersDilemmaTrial(
                        magnitude=monetary_level,
                        monetary_level=monetary_level,
                        low=low,
                        high=high,
                        bonus=bonus,
                        decision=decision,
                        relative_claim=relative_claim,
                        claim_100_scale=claim_100_scale,
                        raw_response=response[:200],
                        trial_number=trial + 1,
                    )
                )

                raw_preview = response.strip().replace("\n", "\\n")
                tqdm.write(
                    f"  Level ${monetary_level:g} ({low}-{high}), Trial {trial + 1}: Raw '{raw_preview[:50]}...' -> Parsed: {decision}"
                )

    def run(self):
        self.run_experiment()
        return self.analyze()

    def analyze(self) -> Dict[str, Any]:
        analysis: Dict[str, Any] = {
            "summary": {},
            "by_magnitude": {},
            "by_monetary_level": {},
        }

        claims = [trial.claim_100_scale for trial in self.trials]
        dollar_claims = [trial.decision for trial in self.trials]
        relative_claims = [trial.relative_claim for trial in self.trials]
        if claims:
            average_claim = float(np.mean(claims))
            analysis["summary"]["overall_average_claim"] = average_claim
            analysis["summary"]["overall_average_claim_100_scale"] = average_claim
            analysis["summary"]["overall_average_claim_dollars"] = float(np.mean(dollar_claims))
            analysis["summary"]["overall_median_claim"] = float(np.median(claims))
            analysis["summary"]["overall_median_claim_dollars"] = float(np.median(dollar_claims))
            analysis["summary"]["overall_normalized_claim"] = float(np.mean(relative_claims))
            lower_bound_rate = sum(1 for trial in self.trials if trial.decision == trial.low) / len(self.trials) * 100
            analysis["summary"]["lower_bound_rate"] = lower_bound_rate

        for monetary_level in self.monetary_levels:
            m_trials = [t for t in self.trials if t.monetary_level == monetary_level]
            if not m_trials:
                continue
            m_claims = [t.claim_100_scale for t in m_trials]
            m_dollar_claims = [t.decision for t in m_trials]
            m_low = m_trials[0].low
            m_high = m_trials[0].high
            level_key = f"{monetary_level:g}"
            
            level_summary = {
                "monetary_level": monetary_level,
                "low": m_low,
                "high": m_high,
                "bonus": m_trials[0].bonus,
                "average_claim": float(np.mean(m_claims)),
                "median_claim": float(np.median(m_claims)),
                "average_claim_dollars": float(np.mean(m_dollar_claims)),
                "median_claim_dollars": float(np.median(m_dollar_claims)),
                "lower_bound_rate": (sum(1 for t in m_trials if t.decision == t.low) / len(m_trials)) * 100,
                "normalized_average_claim": float(np.mean([t.relative_claim for t in m_trials])),
            }
            analysis["by_magnitude"][level_key] = level_summary
            analysis["by_monetary_level"][level_key] = level_summary

        return analysis

    def save_results(self, output_dir: str, model_id: str):
        pd.DataFrame([asdict(trial) for trial in self.trials]).to_csv(
            os.path.join(output_dir, "travellers_dilemma_results.csv"), index=False
        )

        data = {
            "config": {
                "monetary_levels": self.monetary_levels,
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
        avg_claim = analysis["summary"].get("overall_average_claim", 0)
        avg_normalized = analysis["summary"].get("overall_normalized_claim", 0)
        lower_bound_rate = analysis["summary"].get("lower_bound_rate", 0)
        level_breakdown = ", ".join(
            f"${level}: {summary['average_claim']:.1f}/100"
            for level, summary in analysis["by_monetary_level"].items()
        )

        tldr_text = f"Avg Claim: {avg_claim:.1f} / 100."
        analysis_text = f"""
        > DETAILS
        <br><br>
        <b>Overall Strategy:</b> The model chose an average claim of {avg_claim:.1f} on the 2-100 reporting scale across the $10, $100, and $1000 levels.
        <br>
        <b>By Monetary Level:</b> {level_breakdown}.
        <br>
        <b>Equilibrium Pressure:</b> Lower claims are more consistent with the standard unraveling logic of Traveller's Dilemma. The normalized average claim is {avg_normalized:.2f}.
        """

        web_data = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "tldr_text": tldr_text,
            "analysis_text": analysis_text,
            "metrics": {
                "overall_average_claim": avg_claim,
                "overall_average_claim_100_scale": avg_claim,
                "overall_average_claim_dollars": analysis["summary"].get("overall_average_claim_dollars", 0),
                "overall_normalized_claim": avg_normalized,
                "lower_bound_rate": lower_bound_rate,
                "by_magnitude": analysis["by_magnitude"],
                "by_monetary_level": analysis["by_monetary_level"],
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
        for monetary_level in self.monetary_levels:
            m_trials = [t for t in self.trials if t.monetary_level == monetary_level]
            if not m_trials:
                continue
            plot_data.append([t.claim_100_scale for t in m_trials])
            labels.append(f"${monetary_level:g}")
            
        if plot_data:
            plt.boxplot(plot_data, tick_labels=labels)
            plt.xlabel("Monetary Level")
            plt.ylabel("Claim on 2-100 Scale")
            plt.title("Traveller's Dilemma: Claims by Monetary Level")
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
    parser.add_argument(
        "--monetary-levels",
        type=float,
        nargs="+",
        default=MONETARY_LEVELS,
        help="Maximum claim levels to test",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full interactions")
    args = parser.parse_args()

    if args.base_low >= args.base_high:
        print("Error: --base_low must be smaller than --base_high")
        return
    if any(level <= args.base_low for level in args.monetary_levels):
        print("Error: all monetary levels must exceed --base_low")
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
        magnitudes=args.monetary_levels,
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
