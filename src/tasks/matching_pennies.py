"""
Matching Pennies Game Experiment

This script measures an LLM's choice behavior in Matching Pennies.

Template:
You and another person each make a single choice at the same time, without
knowing what the other will choose. You must choose either HEADS or TAILS.
If you both choose the same option, you earn ${WIN_PAYOFF:.2f} and the other
person earns ${LOSE_PAYOFF:.2f}.
If you choose different options, you earn ${LOSE_PAYOFF:.2f} and the other
person earns ${WIN_PAYOFF:.2f}.
What do you choose?
Respond with only "HEADS" or "TAILS".
Your decision:

Default:
You and another person each make a single choice at the same time, without
knowing what the other will choose. You must choose either HEADS or TAILS.
If you both choose the same option, you earn $1.00 and the other person earns
$0.00.
If you choose different options, you earn $0.00 and the other person earns
$1.00.
What do you choose?
Respond with only "HEADS" or "TAILS".
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

WIN_PAYOFFS = [10.0, 100.0, 1000.0]
DEFAULT_WIN_PAYOFF = 1.0
DEFAULT_LOSE_PAYOFF = 0.0


def numeric_key(value: float) -> str:
    return f"{value:g}"


@dataclass
class MatchingPenniesTrial:
    win_payoff: float
    lose_payoff: float
    decision: str  # "HEADS" or "TAILS"
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


def parse_heads_tails(response: str) -> Optional[str]:
    """Parse HEADS or TAILS from model response"""
    response_clean = response.strip()
    response_upper = response_clean.upper()

    if response_upper.startswith("HEADS"):
        return "HEADS"
    if response_upper.startswith("TAILS"):
        return "TAILS"

    labeled_match = re.search(
        r"(?i)(?:choice|decision|answer):\s*(HEADS|TAILS)\b", response_clean
    )
    if labeled_match:
        return labeled_match.group(1).upper()

    matches = re.findall(r"\b(HEADS|TAILS)\b", response_upper)
    if matches:
        return matches[-1]

    return None


# -------------------------------------------------------------
# 4. Prompt Templates
# -------------------------------------------------------------


class MatchingPenniesPrompts:
    @staticmethod
    def template() -> str:
        return """You and another person each make a single choice at the same time, without knowing what the other will choose. You must choose either HEADS or TAILS.
If you both choose the same option, you earn ${WIN_PAYOFF:.2f} and the other person earns ${LOSE_PAYOFF:.2f}.
If you choose different options, you earn ${LOSE_PAYOFF:.2f} and the other person earns ${WIN_PAYOFF:.2f}.
What do you choose?
Respond with only "HEADS" or "TAILS".
Your decision:"""

    @staticmethod
    def default() -> str:
        return MatchingPenniesPrompts.generic_game(
            win_payoff=DEFAULT_WIN_PAYOFF,
            lose_payoff=DEFAULT_LOSE_PAYOFF,
        )

    @staticmethod
    def generic_game(win_payoff: float, lose_payoff: float) -> str:
        return f"""You and another person each make a single choice at the same time, without knowing what the other will choose. You must choose either HEADS or TAILS.
If you both choose the same option, you earn ${win_payoff:.2f} and the other person earns ${lose_payoff:.2f}.
If you choose different options, you earn ${lose_payoff:.2f} and the other person earns ${win_payoff:.2f}.
What do you choose?
Respond with only "HEADS" or "TAILS".
Your decision:"""


# -------------------------------------------------------------
# 5. Experiment Logic
# -------------------------------------------------------------


class MatchingPenniesExperiment:
    def __init__(self, win_payoffs: List[float], lose_payoff: float, n_repetitions: int):
        self.win_payoffs = win_payoffs
        self.lose_payoff = lose_payoff
        self.n_repetitions = n_repetitions
        self.trials: List[MatchingPenniesTrial] = []

    def run_experiment(self):
        print("\nMATCHING PENNIES GAME")
        for win_payoff in self.win_payoffs:
            prompt = MatchingPenniesPrompts.generic_game(
                win_payoff=win_payoff,
                lose_payoff=self.lose_payoff,
            )

            for trial in range(self.n_repetitions):
                response = generate_response(prompt)
                decision = parse_heads_tails(response)

                # Matching Pennies has no pure-strategy equilibrium; default to HEADS if parsing fails.
                if decision is None:
                    decision = "HEADS"

                self.trials.append(
                    MatchingPenniesTrial(
                        win_payoff=win_payoff,
                        lose_payoff=self.lose_payoff,
                        decision=decision,
                        raw_response=response[:200],
                        trial_number=trial + 1,
                    )
                )

                raw_preview = response.strip().replace("\n", "\\n")
                tqdm.write(
                    f"  Win Payoff ${win_payoff:.0f}, Trial {trial + 1}: Raw "
                    f"'{raw_preview[:80]}...' -> Parsed: {decision}"
                )

    def run(self):
        self.run_experiment()
        return self.analyze()

    def analyze(self) -> Dict[str, Any]:
        analysis: Dict[str, Any] = {
            "summary": {},
            "summary_by_win_payoff": {},
            "choice_rates_by_win_payoff": {},
        }

        decisions = [trial.decision for trial in self.trials]
        if decisions:
            heads_count = sum(1 for decision in decisions if decision == "HEADS")
            tails_count = sum(1 for decision in decisions if decision == "TAILS")
            total = len(decisions)

            analysis["summary"]["heads_rate"] = (heads_count / total) * 100
            analysis["summary"]["tails_rate"] = (tails_count / total) * 100
            analysis["summary"]["distance_from_mixed_equilibrium"] = abs(
                (heads_count / total) - 0.5
            ) * 100

        for win_payoff in self.win_payoffs:
            relevant = [trial for trial in self.trials if trial.win_payoff == win_payoff]
            if not relevant:
                continue

            heads_count = sum(1 for trial in relevant if trial.decision == "HEADS")
            tails_count = sum(1 for trial in relevant if trial.decision == "TAILS")
            total = len(relevant)
            payoff_key = numeric_key(win_payoff)
            payoff_summary = {
                "win_payoff": win_payoff,
                "heads_rate": (heads_count / total) * 100,
                "tails_rate": (tails_count / total) * 100,
                "distance_from_mixed_equilibrium": abs((heads_count / total) - 0.5) * 100,
            }
            analysis["summary_by_win_payoff"][payoff_key] = payoff_summary
            analysis["choice_rates_by_win_payoff"][payoff_key] = payoff_summary

        return analysis

    def save_results(self, output_dir: str, model_id: str):
        pd.DataFrame([asdict(trial) for trial in self.trials]).to_csv(
            os.path.join(output_dir, "matching_pennies_results.csv"), index=False
        )

        data = {
            "config": {
                "win_payoffs": self.win_payoffs,
                "lose_payoff": self.lose_payoff,
            },
            "trials": [asdict(trial) for trial in self.trials],
        }
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(data, f, indent=2)

        model_safe = model_id.replace("/", "_").replace(":", "_")
        web_path = os.path.join(
            "web", "data", f"matching_pennies_experiment_{model_safe}.json"
        )

        analysis = self.analyze()
        heads_rate = analysis["summary"].get("heads_rate", 0)
        tails_rate = analysis["summary"].get("tails_rate", 0)
        distance = analysis["summary"].get("distance_from_mixed_equilibrium", 0)

        tldr_text = f"HEADS Rate: {heads_rate:.1f}%."
        analysis_text = f"""
        > DETAILS
        <br><br>
        <b>Overall Strategy:</b> The model chose HEADS {heads_rate:.1f}% of the time and TAILS {tails_rate:.1f}% of the time.
        <br>
        <b>Mixed Strategy Benchmark:</b> Matching Pennies has a 50/50 mixed-strategy equilibrium. The model's HEADS rate was {distance:.1f} percentage points away from 50%.
        """

        web_data = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "win_payoffs": self.win_payoffs,
                "lose_payoff": self.lose_payoff,
            },
            "tldr_text": tldr_text,
            "analysis_text": analysis_text,
            "metrics": {
                "heads_rate": heads_rate,
                "tails_rate": tails_rate,
                "distance_from_mixed_equilibrium": distance,
                "summary_by_win_payoff": analysis["summary_by_win_payoff"],
                "choice_rates_by_win_payoff": analysis["choice_rates_by_win_payoff"],
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
        analysis = self.analyze()
        heads_rates = [
            analysis["summary_by_win_payoff"].get(numeric_key(win_payoff), {}).get("heads_rate", 0)
            for win_payoff in self.win_payoffs
        ]
        tails_rates = [
            analysis["summary_by_win_payoff"].get(numeric_key(win_payoff), {}).get("tails_rate", 0)
            for win_payoff in self.win_payoffs
        ]
        labels = [f"${win_payoff:.0f}" for win_payoff in self.win_payoffs]

        x = range(len(self.win_payoffs))
        width = 0.35

        plt.figure(figsize=(10, 6))
        plt.bar([i - width / 2 for i in x], heads_rates, width, label="HEADS", color="#4682B4", edgecolor="black", alpha=0.85)
        plt.bar([i + width / 2 for i in x], tails_rates, width, label="TAILS", color="#DAA520", edgecolor="black", alpha=0.85)
        plt.axhline(50, color="black", linestyle="--", alpha=0.5, label="50/50 benchmark")
        plt.xticks(list(x), labels)
        plt.ylim(0, 100)
        plt.xlabel("Win Payoff")
        plt.ylabel("Choice Rate (%)")
        plt.title("Matching Pennies: Choice Rates by Payoff")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "choice_rates_by_payoff.png"))
        plt.close()


# -------------------------------------------------------------
# 6. Main
# -------------------------------------------------------------


def main():
    global llm, PRINT_INTERACTIONS

    parser = argparse.ArgumentParser(description="Matching Pennies Game Experiment")
    parser.add_argument("--model", type=str, required=True, help="Model ID")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Number of repetitions for the prompt",
    )
    parser.add_argument(
        "--win-payoff",
        type=float,
        default=None,
        help="Run a single win payoff instead of the default magnitude sweep",
    )
    parser.add_argument(
        "--win-payoffs",
        type=float,
        nargs="+",
        default=None,
        help="Win payoff magnitudes to test",
    )
    parser.add_argument(
        "--lose-payoff",
        type=float,
        default=DEFAULT_LOSE_PAYOFF,
        help="Payoff for the player who loses the round",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full interactions")
    args = parser.parse_args()

    win_payoffs = args.win_payoffs if args.win_payoffs is not None else WIN_PAYOFFS
    if args.win_payoff is not None:
        win_payoffs = [args.win_payoff]

    if any(win_payoff <= args.lose_payoff for win_payoff in win_payoffs):
        print("Error: win payoffs must be greater than --lose-payoff")
        return

    PRINT_INTERACTIONS = args.verbose

    print(f"Initializing model: {args.model}")
    try:
        llm = get_model_interface(args.model)
    except Exception as e:
        print(f"Error loading model {args.model}: {e}")
        return

    output_dir = os.path.join(
        "data", "results", "matching_pennies", args.model.replace("/", "_")
    )
    os.makedirs(output_dir, exist_ok=True)

    exp = MatchingPenniesExperiment(
        win_payoffs=win_payoffs,
        lose_payoff=args.lose_payoff,
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
