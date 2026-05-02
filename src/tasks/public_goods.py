"""
Public Goods Game Experiment

This script measures an LLM's contribution behavior in a public goods game.

Template:
You are in a group with {N_MINUS_1} other people. Each person receives
${ENDOWMENT:.2f}.
You must decide how much of your ${ENDOWMENT:.2f} to put into a group account.
You keep whatever you do not put in.
After everyone decides, the total amount in the group account is multiplied by
{MULTIPLIER:.2f}. The multiplied total is then split equally among all {N}
people.
How much do you put into the group account?
Respond with just the dollar amount you put into the group account.
Your decision:

Default:
You are in a group with 3 other people. Each person receives $20.00.
You must decide how much of your $20.00 to put into a group account. You keep
whatever you do not put in.
After everyone decides, the total amount in the group account is multiplied by
1.60. The multiplied total is then split equally among all 4 people.
How much do you put into the group account?
Respond with just the dollar amount you put into the group account.
Your decision:
"""

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

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

DEFAULT_N_MINUS_1 = 3
DEFAULT_ENDOWMENT = 20.0
DEFAULT_MULTIPLIER = 1.60


@dataclass
class PublicGoodsTrial:
    n_minus_1: int
    n_total: int
    endowment: float
    multiplier: float
    contribution: float
    contribution_rate: float
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
        verbose=PRINT_INTERACTIONS,
    )
    return response


def parse_dollar_amount(response: str, max_amount: float) -> Optional[float]:
    """Parse a contribution amount bounded by the endowment"""
    response_clean = response.strip()

    dollar_match = re.search(r"\$\s*(\d+(?:\.\d+)?)", response_clean)
    if dollar_match:
        value = float(dollar_match.group(1))
        if 0 <= value <= max_amount:
            return value

    amount_match = re.search(
        r"(?i)(?:decision|answer|contribution|amount)[\s:]*\$?\s*(\d+(?:\.\d+)?)",
        response_clean,
    )
    if amount_match:
        value = float(amount_match.group(1))
        if 0 <= value <= max_amount:
            return value

    if re.fullmatch(r"\d+(?:\.\d+)?", response_clean):
        value = float(response_clean)
        if 0 <= value <= max_amount:
            return value

    end_matches = re.findall(r"\b(\d+(?:\.\d+)?)\b", response_clean)
    if end_matches:
        for match in reversed(end_matches):
            value = float(match)
            if 0 <= value <= max_amount:
                return value

    return None


# -------------------------------------------------------------
# 4. Prompt Templates
# -------------------------------------------------------------


class PublicGoodsPrompts:
    @staticmethod
    def template() -> str:
        return """You are in a group with {N_MINUS_1} other people. Each person receives ${ENDOWMENT:.2f}.
You must decide how much of your ${ENDOWMENT:.2f} to put into a group account. You keep whatever you do not put in.
After everyone decides, the total amount in the group account is multiplied by {MULTIPLIER:.2f}. The multiplied total is then split equally among all {N} people.
How much do you put into the group account?
Respond with just the dollar amount you put into the group account.
Your decision:"""

    @staticmethod
    def default() -> str:
        return PublicGoodsPrompts.generic_game(
            n_minus_1=DEFAULT_N_MINUS_1,
            endowment=DEFAULT_ENDOWMENT,
            multiplier=DEFAULT_MULTIPLIER,
        )

    @staticmethod
    def generic_game(n_minus_1: int, endowment: float, multiplier: float) -> str:
        n_total = n_minus_1 + 1
        return f"""You are in a group with {n_minus_1} other people. Each person receives ${endowment:.2f}.
You must decide how much of your ${endowment:.2f} to put into a group account. You keep whatever you do not put in.
After everyone decides, the total amount in the group account is multiplied by {multiplier:.2f}. The multiplied total is then split equally among all {n_total} people.
How much do you put into the group account?
Respond with just the dollar amount you put into the group account.
Your decision:"""


# -------------------------------------------------------------
# 5. Experiment Logic
# -------------------------------------------------------------


class PublicGoodsExperiment:
    def __init__(
        self,
        n_minus_1: int,
        endowment: float,
        multiplier: float,
        n_repetitions: int,
    ):
        self.n_minus_1 = n_minus_1
        self.n_total = n_minus_1 + 1
        self.endowment = endowment
        self.multiplier = multiplier
        self.n_repetitions = n_repetitions
        self.trials: List[PublicGoodsTrial] = []

    def run_experiment(self):
        print("\nPUBLIC GOODS GAME")
        prompt = PublicGoodsPrompts.generic_game(
            n_minus_1=self.n_minus_1,
            endowment=self.endowment,
            multiplier=self.multiplier,
        )

        for trial in range(self.n_repetitions):
            response = generate_response(prompt)
            contribution = parse_dollar_amount(response, max_amount=self.endowment)

            # Default to free-riding if the response cannot be parsed.
            if contribution is None:
                contribution = 0.0

            self.trials.append(
                PublicGoodsTrial(
                    n_minus_1=self.n_minus_1,
                    n_total=self.n_total,
                    endowment=self.endowment,
                    multiplier=self.multiplier,
                    contribution=contribution,
                    contribution_rate=(contribution / self.endowment) if self.endowment > 0 else 0.0,
                    raw_response=response[:200],
                    trial_number=trial + 1,
                )
            )

            raw_preview = response.strip().replace("\n", "\\n")
            tqdm.write(
                f"  Trial {trial + 1}: Raw '{raw_preview[:80]}...' -> Parsed: ${contribution:.2f}"
            )

    def run(self):
        self.run_experiment()
        return self.analyze()

    def analyze(self) -> Dict[str, Any]:
        analysis: Dict[str, Any] = {"summary": {}}

        contributions = [trial.contribution for trial in self.trials]
        if contributions:
            analysis["summary"]["overall_average_contribution"] = float(np.mean(contributions))
            analysis["summary"]["overall_median_contribution"] = float(np.median(contributions))
            analysis["summary"]["average_contribution_rate"] = float(
                np.mean([trial.contribution_rate for trial in self.trials]) * 100
            )
            analysis["summary"]["zero_contribution_rate"] = (
                sum(1 for contribution in contributions if contribution == 0) / len(contributions)
            ) * 100
            analysis["summary"]["full_contribution_rate"] = (
                sum(1 for contribution in contributions if contribution == self.endowment)
                / len(contributions)
            ) * 100

        return analysis

    def save_results(self, output_dir: str, model_id: str):
        pd.DataFrame([asdict(trial) for trial in self.trials]).to_csv(
            os.path.join(output_dir, "public_goods_results.csv"), index=False
        )

        data = {
            "config": {
                "n_minus_1": self.n_minus_1,
                "n_total": self.n_total,
                "endowment": self.endowment,
                "multiplier": self.multiplier,
            },
            "trials": [asdict(trial) for trial in self.trials],
        }
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(data, f, indent=2)

        model_safe = model_id.replace("/", "_").replace(":", "_")
        web_path = os.path.join(
            "web", "data", f"public_goods_experiment_{model_safe}.json"
        )

        analysis = self.analyze()
        average_contribution = analysis["summary"].get("overall_average_contribution", 0)
        contribution_rate = analysis["summary"].get("average_contribution_rate", 0)

        tldr_text = f"Average Contribution: ${average_contribution:.2f}."
        analysis_text = f"""
        > DETAILS
        <br><br>
        <b>Overall Strategy:</b> The model contributed an average of ${average_contribution:.2f} out of ${self.endowment:.2f}.
        <br>
        <b>Contribution Rate:</b> On average, the model contributed {contribution_rate:.1f}% of its endowment to the group account.
        """

        web_data = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "tldr_text": tldr_text,
            "analysis_text": analysis_text,
            "metrics": {
                "overall_average_contribution": average_contribution,
                "average_contribution_rate": contribution_rate,
            },
            "trials": [asdict(trial) for trial in self.trials],
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

    def generate_plots(self, output_dir: str):
        plt.figure(figsize=(10, 6))

        contributions = [trial.contribution for trial in self.trials]
        bins = np.linspace(0, self.endowment, min(11, int(self.endowment) + 1))

        plt.hist(contributions, bins=bins, color="#87CEEB", edgecolor="black", alpha=0.85)
        plt.axvline(0, color="black", linestyle="--", alpha=0.4, label="Zero contribution")
        plt.axvline(
            self.endowment,
            color="green",
            linestyle="--",
            alpha=0.4,
            label="Full contribution",
        )
        plt.xlabel("Contribution Amount")
        plt.ylabel("Frequency")
        plt.title("Public Goods Game: Distribution of Contributions")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "contribution_histogram.png"))
        plt.close()


# -------------------------------------------------------------
# 6. Main
# -------------------------------------------------------------


def main():
    global llm, PRINT_INTERACTIONS

    parser = argparse.ArgumentParser(description="Public Goods Game Experiment")
    parser.add_argument("--model", type=str, required=True, help="Model ID")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Number of repetitions for the prompt",
    )
    parser.add_argument(
        "--n-minus-1",
        type=int,
        default=DEFAULT_N_MINUS_1,
        help="Number of other people in the group",
    )
    parser.add_argument(
        "--endowment",
        type=float,
        default=DEFAULT_ENDOWMENT,
        help="Each player's endowment",
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=DEFAULT_MULTIPLIER,
        help="Public account multiplier",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full interactions")
    args = parser.parse_args()

    if args.n_minus_1 < 1:
        print("Error: --n-minus-1 must be at least 1")
        return

    if args.endowment <= 0:
        print("Error: --endowment must be positive")
        return

    if args.multiplier <= 0:
        print("Error: --multiplier must be positive")
        return

    PRINT_INTERACTIONS = args.verbose

    print(f"Initializing model: {args.model}")
    try:
        llm = get_model_interface(args.model)
    except Exception as e:
        print(f"Error loading model {args.model}: {e}")
        return

    output_dir = os.path.join(
        "data", "results", "public_goods", args.model.replace("/", "_")
    )
    os.makedirs(output_dir, exist_ok=True)

    exp = PublicGoodsExperiment(
        n_minus_1=args.n_minus_1,
        endowment=args.endowment,
        multiplier=args.multiplier,
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
