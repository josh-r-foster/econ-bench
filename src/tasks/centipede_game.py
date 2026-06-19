"""
Centipede Game Experiment

This script measures an LLM's willingness to continue cooperating in a finite
centipede game.

Template:
You are playing a game with another person. You take turns choosing PASS or
TAKE. If either player chooses TAKE, the game ends immediately. If a player
chooses PASS, the other player gets the next turn. If both players keep
passing, the game ends after the last turn.
Here is what happens at each turn:
{GAME_TREE}
It is currently your turn ({CURRENT_TURN_LABEL}).
Do you choose PASS or TAKE?
Respond with only "PASS" or "TAKE".
Your decision:

Default:
You are playing a game with another person. You take turns choosing PASS or
TAKE. If either player chooses TAKE, the game ends immediately. If a player
chooses PASS, the other player gets the next turn. If both players keep
passing, the game ends after the last turn.
Here is what happens at each turn:
Turn 1 (you): TAKE -> you earn $2.00, they earn $1.00.
Turn 2 (them): TAKE -> you earn $1.00, they earn $4.00.
Turn 3 (you): TAKE -> you earn $4.00, they earn $2.00.
Turn 4 (them): TAKE -> you earn $2.00, they earn $8.00.
Turn 5 (you): TAKE -> you earn $8.00, they earn $4.00.
Turn 6 (them): TAKE -> you earn $4.00, they earn $16.00.
If no one takes by Turn 6: you earn $16.00, they earn $8.00.
It is currently your turn (Turn 1).
Do you choose PASS or TAKE?
Respond with only "PASS" or "TAKE".
Your decision:
"""

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


@dataclass(frozen=True)
class CentipedeTurn:
    turn_number: int
    player: str
    take_payoff_you: float
    take_payoff_them: float


DEFAULT_TURNS = [
    CentipedeTurn(1, "you", 1.25, 0.62),
    CentipedeTurn(2, "them", 0.62, 2.50),
    CentipedeTurn(3, "you", 2.50, 1.25),
    CentipedeTurn(4, "them", 1.25, 5.00),
    CentipedeTurn(5, "you", 5.00, 2.50),
    CentipedeTurn(6, "them", 2.50, 10.00),
]
DEFAULT_FINAL_PAYOFFS = (10.00, 5.00)
DEFAULT_CURRENT_TURN = 1
MAGNITUDES = [10.0, 100.0, 1000.0]

def generate_turns(magnitude: float) -> Tuple[List[CentipedeTurn], Tuple[float, float]]:
    scaled_turns = [
        CentipedeTurn(
            turn.turn_number, 
            turn.player, 
            turn.take_payoff_you * magnitude, 
            turn.take_payoff_them * magnitude
        ) for turn in DEFAULT_TURNS
    ]
    scaled_final = (DEFAULT_FINAL_PAYOFFS[0] * magnitude, DEFAULT_FINAL_PAYOFFS[1] * magnitude)
    return scaled_turns, scaled_final

@dataclass
class CentipedeTrial:
    magnitude: int
    current_turn: int
    current_turn_label: str
    take_payoff_you: float
    take_payoff_them: float
    decision: str  # "PASS" or "TAKE"
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


def parse_pass_take(response: str) -> Optional[str]:
    """Parse PASS or TAKE from model response"""
    response_clean = response.strip()
    response_upper = response_clean.upper()

    # 1. Exact match or starts with
    if response_upper.startswith("PASS"):
        return "PASS"
    if response_upper.startswith("TAKE"):
        return "TAKE"

    # 2. Look for a labeled answer like "Your decision: TAKE"
    labeled_match = re.search(
        r"(?i)(?:choice|decision|answer):\s*(PASS|TAKE)\b", response_clean
    )
    if labeled_match:
        return labeled_match.group(1).upper()

    # 3. Use the last standalone PASS/TAKE token in the response
    matches = re.findall(r"\b(PASS|TAKE)\b", response_upper)
    if matches:
        return matches[-1]

    return None


def format_turn_label(turn_number: int) -> str:
    return f"Turn {turn_number}"


def format_game_tree(
    turns: List[CentipedeTurn], final_payoffs: Tuple[float, float]
) -> str:
    lines = []
    for turn in turns:
        lines.append(
            f"Turn {turn.turn_number} ({turn.player}): TAKE -> you earn "
            f"${turn.take_payoff_you:.2f}, they earn ${turn.take_payoff_them:.2f}."
        )

    last_turn = turns[-1].turn_number
    lines.append(
        f"If no one takes by Turn {last_turn}: you earn ${final_payoffs[0]:.2f}, "
        f"they earn ${final_payoffs[1]:.2f}."
    )
    return "\n".join(lines)


# -------------------------------------------------------------
# 4. Prompt Templates
# -------------------------------------------------------------


class CentipedeGamePrompts:
    @staticmethod
    def template() -> str:
        return """You are playing a game with another person. You take turns choosing PASS or TAKE. If either player chooses TAKE, the game ends immediately. If a player chooses PASS, the other player gets the next turn. If both players keep passing, the game ends after the last turn.
Here is what happens at each turn:
{GAME_TREE}
It is currently your turn ({CURRENT_TURN_LABEL}).
Do you choose PASS or TAKE?
Respond with only "PASS" or "TAKE".
Your decision:"""

    @staticmethod
    def default() -> str:
        return CentipedeGamePrompts.generic_game(
            game_tree=format_game_tree(DEFAULT_TURNS, DEFAULT_FINAL_PAYOFFS),
            current_turn_label=format_turn_label(DEFAULT_CURRENT_TURN),
        )

    @staticmethod
    def generic_game(game_tree: str, current_turn_label: str) -> str:
        return f"""You are playing a game with another person. You take turns choosing PASS or TAKE. If either player chooses TAKE, the game ends immediately. If a player chooses PASS, the other player gets the next turn. If both players keep passing, the game ends after the last turn.
Here is what happens at each turn:
{game_tree}
It is currently your turn ({current_turn_label}).
Do you choose PASS or TAKE?
Respond with only "PASS" or "TAKE".
Your decision:"""


# -------------------------------------------------------------
# 5. Experiment Logic
# -------------------------------------------------------------


class CentipedeGameExperiment:
    def __init__(
        self,
        magnitudes: List[int],
        n_repetitions: int,
    ):
        self.magnitudes = magnitudes
        self.n_repetitions = n_repetitions
        self.trials: List[CentipedeTrial] = []
        self.query_turns = [turn for turn in DEFAULT_TURNS if turn.player == "you"]

    def run_experiment(self):
        print("\nCENTIPEDE GAME")
        
        for magnitude in self.magnitudes:
            turns, final_payoffs = generate_turns(magnitude)
            game_tree = format_game_tree(turns, final_payoffs)

            for q_turn in self.query_turns:
                current_turn_label = format_turn_label(q_turn.turn_number)
                scaled_turn = next(t for t in turns if t.turn_number == q_turn.turn_number)
                
                for trial in range(self.n_repetitions):
                    prompt = CentipedeGamePrompts.generic_game(
                        game_tree=game_tree,
                        current_turn_label=current_turn_label,
                    )
                    response = generate_response(prompt)

                    decision = parse_pass_take(response) or "TAKE"

                    self.trials.append(
                        CentipedeTrial(
                            magnitude=magnitude,
                            current_turn=q_turn.turn_number,
                            current_turn_label=current_turn_label,
                            take_payoff_you=scaled_turn.take_payoff_you,
                            take_payoff_them=scaled_turn.take_payoff_them,
                            decision=decision,
                            raw_response=response[:200],
                            trial_number=trial + 1,
                        )
                    )

                    raw_preview = response.strip().replace("\n", "\\n")
                    tqdm.write(
                        f"  Magnitude {magnitude}x, {current_turn_label}, Trial {trial + 1}: Raw "
                        f"'{raw_preview[:50]}...' -> Parsed: {decision}"
                    )

    def run(self):
        self.run_experiment()
        return self.analyze()

    def analyze(self) -> Dict[str, Any]:
        analysis: Dict[str, Any] = {
            "summary": {},
            "by_magnitude": {}
        }

        all_decisions = [trial.decision for trial in self.trials]
        if all_decisions:
            take_count = sum(1 for decision in all_decisions if decision == "TAKE")
            pass_count = sum(1 for decision in all_decisions if decision == "PASS")
            analysis["summary"]["overall_take_rate"] = (take_count / len(all_decisions)) * 100
            analysis["summary"]["overall_pass_rate"] = (pass_count / len(all_decisions)) * 100
            analysis["summary"]["backward_induction_consistency"] = (
                take_count / len(all_decisions)
            ) * 100

        for magnitude in self.magnitudes:
            m_trials = [t for t in self.trials if t.magnitude == magnitude]
            if not m_trials:
                continue
                
            m_analysis = {
                "take_rate_by_turn": {},
                "pass_rate_by_turn": {}
            }
            
            for turn in self.query_turns:
                relevant = [trial for trial in m_trials if trial.current_turn == turn.turn_number]
                if not relevant:
                    continue

                take_count = sum(1 for trial in relevant if trial.decision == "TAKE")
                pass_count = sum(1 for trial in relevant if trial.decision == "PASS")
                m_analysis["take_rate_by_turn"][turn.turn_number] = (take_count / len(relevant)) * 100
                m_analysis["pass_rate_by_turn"][turn.turn_number] = (pass_count / len(relevant)) * 100
                
            analysis["by_magnitude"][magnitude] = m_analysis

        return analysis

    def save_results(self, output_dir: str, model_id: str):
        pd.DataFrame([asdict(trial) for trial in self.trials]).to_csv(
            os.path.join(output_dir, "centipede_game_results.csv"), index=False
        )

        data = {
            "config": {
                "magnitudes": self.magnitudes,
                "base_turns": [asdict(turn) for turn in DEFAULT_TURNS],
                "base_final_payoffs": {
                    "you": DEFAULT_FINAL_PAYOFFS[0],
                    "them": DEFAULT_FINAL_PAYOFFS[1],
                },
            },
            "trials": [asdict(trial) for trial in self.trials],
        }
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(data, f, indent=2)

        model_safe = model_id.replace("/", "_").replace(":", "_")
        web_path = os.path.join(
            "web", "data", f"centipede_game_experiment_{model_safe}.json"
        )

        analysis = self.analyze()
        overall_take = analysis["summary"].get("overall_take_rate", 0)

        tldr_text = f"Take Rate: {overall_take:.1f}%."
        analysis_text = f"""
        > DETAILS
        <br><br>
        <b>Overall Strategy:</b> The model chose TAKE {overall_take:.1f}% of the time.
        <br>
        <b>Interpretation:</b> In a finite centipede game, backward induction predicts TAKE at the earliest opportunity. Higher PASS rates indicate more willingness to continue the cooperative path.
        """

        web_data = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "tldr_text": tldr_text,
            "analysis_text": analysis_text,
            "metrics": {
                "overall_take_rate": overall_take,
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
        import numpy as np
        plt.figure(figsize=(12, 6))

        turn_numbers = [turn.turn_number for turn in self.query_turns]
        analysis = self.analyze()
        
        x = np.arange(len(turn_numbers))
        width = 0.8 / len(self.magnitudes)
        
        for i, magnitude in enumerate(self.magnitudes):
            m_analysis = analysis["by_magnitude"].get(magnitude, {})
            take_rates = [
                m_analysis.get("take_rate_by_turn", {}).get(turn_number, 0)
                for turn_number in turn_numbers
            ]
            
            offset = (i - len(self.magnitudes)/2 + 0.5) * width
            plt.bar(
                x + offset,
                take_rates,
                width,
                label=f"{magnitude}x Scale"
            )

        plt.xticks(x, [format_turn_label(turn_number) for turn_number in turn_numbers])
        plt.ylim(0, 100)
        plt.xlabel("Decision Turn")
        plt.ylabel("Take Rate (%)")
        plt.title("Centipede Game: TAKE Rate by Turn across Magnitudes")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "take_rate_by_turn.png"))
        plt.close()


# -------------------------------------------------------------
# 6. Main
# -------------------------------------------------------------


def main():
    global llm, PRINT_INTERACTIONS

    parser = argparse.ArgumentParser(description="Centipede Game Experiment")
    parser.add_argument("--model", type=str, required=True, help="Model ID")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Number of repetitions per queried turn",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full interactions")
    args = parser.parse_args()

    PRINT_INTERACTIONS = args.verbose

    print(f"Initializing model: {args.model}")
    try:
        llm = get_model_interface(args.model)
    except Exception as e:
        print(f"Error loading model {args.model}: {e}")
        return

    output_dir = os.path.join(
        "data", "results", "centipede_game", args.model.replace("/", "_")
    )
    os.makedirs(output_dir, exist_ok=True)

    exp = CentipedeGameExperiment(
        magnitudes=MAGNITUDES,
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
