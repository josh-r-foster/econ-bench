"""Traveller's Dilemma with proportionally scaled monetary treatments."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from dataclasses import asdict, dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.results.model_ids import model_id_to_path_component
from src.results.provenance import utc_now
from src.tasks.response_formats import parse_bounded_amount
from src.tasks.runtime import request_model_response

# -------------------------------------------------------------
# 1. Configuration & Global State
# -------------------------------------------------------------

# -------------------------------------------------------------
# 2. Experimental Parameters & Data Structures
# -------------------------------------------------------------

MONETARY_LEVELS = [10.0, 100.0, 1000.0]
MAGNITUDES = MONETARY_LEVELS
BASE_LOW = 2
BASE_HIGH = 100
BASE_BONUS = 2
BASE_INCREMENT = 1
WEB_LOW = 2
WEB_HIGH = 100


@dataclass
class TravellersDilemmaTrial:
    magnitude: float
    monetary_level: float
    low: float
    high: float
    bonus: float
    decision: float
    relative_claim: float
    claim_100_scale: float
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
        experiment_id="travellers_dilemma",
        prompt=prompt,
        max_new_tokens=8192,
        temperature=temperature,
        verbose=verbose,
    )


def parse_number(
    response: str,
    low: float,
    high: float,
    increment: float = BASE_INCREMENT,
) -> Optional[float]:
    """Parse a feasible claim on the scaled action grid."""
    return parse_bounded_amount(
        response,
        minimum=low,
        maximum=high,
        increment=increment,
        allow_percentage=False,
        labels=("claim",),
    )


def monetary_bounds_for_level(
    monetary_level: float,
    base_low: float,
    base_high: float,
    base_bonus: float,
) -> Tuple[float, float, float]:
    high = float(monetary_level)
    scale = high / base_high
    low = base_low * scale
    bonus = base_bonus * scale

    if low >= high:
        raise ValueError(f"Monetary level ${monetary_level:g} must exceed lower bound {low}")

    return low, high, bonus


def monetary_increment_for_level(
    monetary_level: float,
    base_high: float,
    base_increment: float = BASE_INCREMENT,
) -> float:
    """Scale the claim grid with the payoff treatment."""
    return base_increment * monetary_level / base_high


def claim_on_100_scale(relative_claim: float) -> float:
    return WEB_LOW + relative_claim * (WEB_HIGH - WEB_LOW)


# -------------------------------------------------------------
# 4. Prompt Templates
# -------------------------------------------------------------


class TravellersDilemmaPrompts:
    @staticmethod
    def template() -> str:
        return """You and another person simultaneously choose a dollar claim from ${LOW} to ${HIGH} in increments of ${INCREMENT}. Neither person sees the other's claim before choosing.
If both claims are equal, each person receives that amount.
If the claims differ, the person making the lower claim receives the lower claim plus ${BONUS}. The person making the higher claim receives the lower claim minus ${BONUS}.
What amount do you claim?
Return one line in the form CLAIM=<amount> with at most two digits after the decimal point.
Your claim"""

    @staticmethod
    def default() -> str:
        return TravellersDilemmaPrompts.generic_game(
            low=BASE_LOW,
            high=BASE_HIGH,
            bonus=BASE_BONUS,
            increment=BASE_INCREMENT,
        )

    @staticmethod
    def generic_game(
        low: float, high: float, bonus: float, increment: float = BASE_INCREMENT
    ) -> str:
        return f"""You and another person simultaneously choose a dollar claim from ${low:.2f} to ${high:.2f} in increments of ${increment:.2f}. Neither person sees the other's claim before choosing.
If both claims are equal, each person receives that amount.
If the claims differ, the person making the lower claim receives the lower claim plus ${bonus:.2f}. The person making the higher claim receives the lower claim minus ${bonus:.2f}.
What amount do you claim?
Return one line in the form CLAIM=<amount> with at most two digits after the decimal point.
Your claim"""


# -------------------------------------------------------------
# 5. Experiment Logic
# -------------------------------------------------------------


class TravellersDilemmaExperiment:
    def __init__(self, magnitudes: List[float], base_low: int, base_high: int,
                 base_bonus: int, n_repetitions: int, interface=None,
                 verbose: bool = False):
        self.magnitudes = magnitudes
        self.monetary_levels = magnitudes
        self.base_low = base_low
        self.base_high = base_high
        self.base_bonus = base_bonus
        self.n_repetitions = n_repetitions
        self.interface = interface
        self.verbose = verbose
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
            increment = monetary_increment_for_level(
                monetary_level, self.base_high
            )
            
            prompt = TravellersDilemmaPrompts.generic_game(
                low=low,
                high=high,
                bonus=bonus,
                increment=increment,
            )

            for trial in range(self.n_repetitions):
                response = generate_response(self.interface, prompt, verbose=self.verbose)
                decision = parse_number(
                    response, low=low, high=high, increment=increment
                )

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
        relative_claims = [trial.relative_claim for trial in self.trials]
        if claims:
            average_claim = float(np.mean(claims))
            analysis["summary"]["overall_average_claim"] = average_claim
            analysis["summary"]["overall_average_claim_100_scale"] = average_claim
            analysis["summary"]["overall_median_claim"] = float(np.median(claims))
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

        model_key = model_id_to_path_component(model_id)
        web_path = os.path.join(
            "web", "data", f"travellers_dilemma_experiment_{model_key}.json"
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
            "timestamp": utc_now(),
            "tldr_text": tldr_text,
            "analysis_text": analysis_text,
            "metrics": {
                "overall_average_claim": avg_claim,
                "overall_average_claim_100_scale": avg_claim,
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
    from src.tasks.engine import run_single_experiment_cli
    return run_single_experiment_cli("travellers_dilemma")


if __name__ == "__main__":
    raise SystemExit(main())
