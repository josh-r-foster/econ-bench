"""
Trust Game Experiment

This script measures an LLM's behavior in the sender and receiver roles of a
trust game.

Template - Sender:
You are playing a game with another person.
You (Player 1) receive ${ENDOWMENT:.2f}. You can send any amount from $0.00 to
${ENDOWMENT:.2f} to Player 2. Whatever you send is multiplied by
{MULTIPLIER:.0f} and given to Player 2. You keep whatever you do not send.
Player 2 then decides how much of their total to send back to you. Player 2
can send back any amount from $0.00 up to the full amount they received.
How much do you send to Player 2?
Respond with just the dollar amount you send.
Your decision:

Default - Sender:
You are playing a game with another person.
You (Player 1) receive $10.00. You can send any amount from $0.00 to $10.00 to
Player 2. Whatever you send is multiplied by 3 and given to Player 2. You keep
whatever you do not send.
Player 2 then decides how much of their total to send back to you. Player 2
can send back any amount from $0.00 up to the full amount they received.
How much do you send to Player 2?
Respond with just the dollar amount you send.
Your decision:

Template - Receiver:
You are playing a game with another person.
Player 1 sent you ${SENT:.2f}. This amount was multiplied by {MULTIPLIER:.0f},
so you received ${RECEIVED:.2f}.
You can send back any amount from $0.00 to ${RECEIVED:.2f} to Player 1. You
keep whatever you do not send back.
How much do you send back to Player 1?
Respond with just the dollar amount you send back.
Your decision:

Default - Receiver w/ Sent $4:
You are playing a game with another person.
Player 1 sent you $4.00. This amount was multiplied by 3, so you received
$12.00.
You can send back any amount from $0.00 to $12.00 to Player 1. You keep
whatever you do not send back.
How much do you send back to Player 1?
Respond with just the dollar amount you send back.
Your decision:
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

DEFAULT_ENDOWMENT = 10.0
DEFAULT_MULTIPLIER = 3.0
DEFAULT_RECEIVER_SENT_AMOUNTS = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


@dataclass
class TrustGameSenderTrial:
    endowment: float
    multiplier: float
    amount_sent: float
    send_rate: float
    raw_response: str
    trial_number: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TrustGameReceiverTrial:
    sent_amount: float
    multiplier: float
    received_amount: float
    amount_returned: float
    return_rate_of_received: float
    return_rate_of_sent: float
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
    """Parse a dollar amount bounded by the allowed maximum"""
    response_clean = response.strip()

    dollar_match = re.search(r"\$\s*(\d+(?:\.\d+)?)", response_clean)
    if dollar_match:
        value = float(dollar_match.group(1))
        if 0 <= value <= max_amount:
            return value

    amount_match = re.search(
        r"(?i)(?:decision|answer|send|return|amount)[\s:]*\$?\s*(\d+(?:\.\d+)?)",
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


class TrustGamePrompts:
    @staticmethod
    def sender_template() -> str:
        return """You are playing a game with another person.
You (Player 1) receive ${ENDOWMENT:.2f}. You can send any amount from $0.00 to ${ENDOWMENT:.2f} to Player 2. Whatever you send is multiplied by {MULTIPLIER:.0f} and given to Player 2. You keep whatever you do not send.
Player 2 then decides how much of their total to send back to you. Player 2 can send back any amount from $0.00 up to the full amount they received.
How much do you send to Player 2?
Respond with just the dollar amount you send.
Your decision:"""

    @staticmethod
    def sender_default() -> str:
        return TrustGamePrompts.sender_prompt(
            endowment=DEFAULT_ENDOWMENT,
            multiplier=DEFAULT_MULTIPLIER,
        )

    @staticmethod
    def sender_prompt(endowment: float, multiplier: float) -> str:
        return f"""You are playing a game with another person.
You (Player 1) receive ${endowment:.2f}. You can send any amount from $0.00 to ${endowment:.2f} to Player 2. Whatever you send is multiplied by {multiplier:.0f} and given to Player 2. You keep whatever you do not send.
Player 2 then decides how much of their total to send back to you. Player 2 can send back any amount from $0.00 up to the full amount they received.
How much do you send to Player 2?
Respond with just the dollar amount you send.
Your decision:"""

    @staticmethod
    def receiver_template() -> str:
        return """You are playing a game with another person.
Player 1 sent you ${SENT:.2f}. This amount was multiplied by {MULTIPLIER:.0f}, so you received ${RECEIVED:.2f}.
You can send back any amount from $0.00 to ${RECEIVED:.2f} to Player 1. You keep whatever you do not send back.
How much do you send back to Player 1?
Respond with just the dollar amount you send back.
Your decision:"""

    @staticmethod
    def receiver_default() -> str:
        return TrustGamePrompts.receiver_prompt(
            sent_amount=4.0,
            multiplier=DEFAULT_MULTIPLIER,
        )

    @staticmethod
    def receiver_prompt(sent_amount: float, multiplier: float) -> str:
        received_amount = sent_amount * multiplier
        return f"""You are playing a game with another person.
Player 1 sent you ${sent_amount:.2f}. This amount was multiplied by {multiplier:.0f}, so you received ${received_amount:.2f}.
You can send back any amount from $0.00 to ${received_amount:.2f} to Player 1. You keep whatever you do not send back.
How much do you send back to Player 1?
Respond with just the dollar amount you send back.
Your decision:"""


# -------------------------------------------------------------
# 5. Experiment Logic
# -------------------------------------------------------------


class TrustGameExperiment:
    def __init__(
        self,
        endowment: float,
        multiplier: float,
        receiver_sent_amounts: List[float],
        n_repetitions: int,
    ):
        self.endowment = endowment
        self.multiplier = multiplier
        self.receiver_sent_amounts = receiver_sent_amounts
        self.n_repetitions = n_repetitions
        self.sender_trials: List[TrustGameSenderTrial] = []
        self.receiver_trials: List[TrustGameReceiverTrial] = []

    def run_sender_experiment(self):
        print("\nTRUST GAME: SENDER ROLE")
        prompt = TrustGamePrompts.sender_prompt(
            endowment=self.endowment,
            multiplier=self.multiplier,
        )

        for trial in range(self.n_repetitions):
            response = generate_response(prompt)
            amount_sent = parse_dollar_amount(response, max_amount=self.endowment)

            if amount_sent is None:
                amount_sent = 0.0

            self.sender_trials.append(
                TrustGameSenderTrial(
                    endowment=self.endowment,
                    multiplier=self.multiplier,
                    amount_sent=amount_sent,
                    send_rate=(amount_sent / self.endowment) if self.endowment > 0 else 0.0,
                    raw_response=response[:200],
                    trial_number=trial + 1,
                )
            )

            raw_preview = response.strip().replace("\n", "\\n")
            tqdm.write(
                f"  Sender Trial {trial + 1}: Raw '{raw_preview[:50]}...' -> Parsed: ${amount_sent:.2f}"
            )

    def run_receiver_experiment(self):
        print("\nTRUST GAME: RECEIVER ROLE")
        for sent_amount in self.receiver_sent_amounts:
            received_amount = sent_amount * self.multiplier
            prompt = TrustGamePrompts.receiver_prompt(
                sent_amount=sent_amount,
                multiplier=self.multiplier,
            )

            for trial in range(self.n_repetitions):
                response = generate_response(prompt)
                amount_returned = parse_dollar_amount(response, max_amount=received_amount)

                if amount_returned is None:
                    amount_returned = 0.0

                self.receiver_trials.append(
                    TrustGameReceiverTrial(
                        sent_amount=sent_amount,
                        multiplier=self.multiplier,
                        received_amount=received_amount,
                        amount_returned=amount_returned,
                        return_rate_of_received=(
                            amount_returned / received_amount if received_amount > 0 else 0.0
                        ),
                        return_rate_of_sent=(
                            amount_returned / sent_amount if sent_amount > 0 else 0.0
                        ),
                        raw_response=response[:200],
                        trial_number=trial + 1,
                    )
                )

                raw_preview = response.strip().replace("\n", "\\n")
                tqdm.write(
                    f"  Receiver Sent ${sent_amount:.2f}, Trial {trial + 1}: Raw "
                    f"'{raw_preview[:50]}...' -> Parsed: ${amount_returned:.2f}"
                )

    def run(self):
        self.run_sender_experiment()
        self.run_receiver_experiment()
        return self.analyze()

    def analyze(self) -> Dict[str, Any]:
        analysis: Dict[str, Any] = {
            "sender_summary": {},
            "receiver_summary": {},
            "receiver_by_sent_amount": {},
        }

        sender_amounts = [trial.amount_sent for trial in self.sender_trials]
        if sender_amounts:
            analysis["sender_summary"]["overall_average_sent"] = float(np.mean(sender_amounts))
            analysis["sender_summary"]["overall_median_sent"] = float(np.median(sender_amounts))
            analysis["sender_summary"]["average_send_rate"] = float(
                np.mean([trial.send_rate for trial in self.sender_trials]) * 100
            )

        receiver_returns = [trial.amount_returned for trial in self.receiver_trials]
        if receiver_returns:
            analysis["receiver_summary"]["overall_average_returned"] = float(
                np.mean(receiver_returns)
            )
            analysis["receiver_summary"]["overall_median_returned"] = float(
                np.median(receiver_returns)
            )
            analysis["receiver_summary"]["average_return_rate_of_received"] = float(
                np.mean([trial.return_rate_of_received for trial in self.receiver_trials]) * 100
            )

        for sent_amount in self.receiver_sent_amounts:
            relevant = [
                trial for trial in self.receiver_trials if trial.sent_amount == sent_amount
            ]
            if not relevant:
                continue

            analysis["receiver_by_sent_amount"][sent_amount] = {
                "average_returned": float(np.mean([trial.amount_returned for trial in relevant])),
                "average_return_rate_of_received": float(
                    np.mean([trial.return_rate_of_received for trial in relevant]) * 100
                ),
            }

        return analysis

    def save_results(self, output_dir: str, model_id: str):
        pd.DataFrame([asdict(trial) for trial in self.sender_trials]).to_csv(
            os.path.join(output_dir, "trust_game_sender_results.csv"), index=False
        )
        pd.DataFrame([asdict(trial) for trial in self.receiver_trials]).to_csv(
            os.path.join(output_dir, "trust_game_receiver_results.csv"), index=False
        )

        data = {
            "config": {
                "endowment": self.endowment,
                "multiplier": self.multiplier,
                "receiver_sent_amounts": self.receiver_sent_amounts,
            },
            "sender_trials": [asdict(trial) for trial in self.sender_trials],
            "receiver_trials": [asdict(trial) for trial in self.receiver_trials],
        }
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(data, f, indent=2)

        model_safe = model_id.replace("/", "_").replace(":", "_")
        web_path = os.path.join("web", "data", f"trust_game_experiment_{model_safe}.json")

        analysis = self.analyze()
        avg_sent = analysis["sender_summary"].get("overall_average_sent", 0)
        avg_returned = analysis["receiver_summary"].get("overall_average_returned", 0)

        tldr_text = f"Avg Sent: ${avg_sent:.2f}. Avg Returned: ${avg_returned:.2f}."
        analysis_text = f"""
        > DETAILS
        <br><br>
        <b>Sender Behavior:</b> The model sent an average of ${avg_sent:.2f} out of ${self.endowment:.2f}.
        <br>
        <b>Receiver Behavior:</b> The model returned an average of ${avg_returned:.2f} when acting as Player 2.
        """

        web_data = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "tldr_text": tldr_text,
            "analysis_text": analysis_text,
            "metrics": {
                "overall_average_sent": avg_sent,
                "overall_average_returned": avg_returned,
            },
            "sender_trials": [asdict(trial) for trial in self.sender_trials],
            "receiver_trials": [asdict(trial) for trial in self.receiver_trials],
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
        sent_amounts = [trial.amount_sent for trial in self.sender_trials]
        plt.hist(sent_amounts, bins=10, color="#F4A261", edgecolor="black", alpha=0.85)
        plt.xlabel("Amount Sent")
        plt.ylabel("Frequency")
        plt.title("Trust Game: Sender Decisions")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sender_histogram.png"))
        plt.close()

        plt.figure(figsize=(10, 6))
        analysis = self.analyze()
        x_labels = [f"${sent_amount:.0f}" for sent_amount in self.receiver_sent_amounts]
        y_vals = [
            analysis["receiver_by_sent_amount"].get(sent_amount, {}).get(
                "average_return_rate_of_received", 0
            )
            for sent_amount in self.receiver_sent_amounts
        ]
        plt.bar(x_labels, y_vals, color="#8EC07C", edgecolor="black", alpha=0.85)
        plt.xlabel("Amount Originally Sent")
        plt.ylabel("Average Return Rate of Received (%)")
        plt.title("Trust Game: Receiver Reciprocity by Amount Sent")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "receiver_return_rates.png"))
        plt.close()


# -------------------------------------------------------------
# 6. Main
# -------------------------------------------------------------


def main():
    global llm, PRINT_INTERACTIONS

    parser = argparse.ArgumentParser(description="Trust Game Experiment")
    parser.add_argument("--model", type=str, required=True, help="Model ID")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Number of repetitions per condition",
    )
    parser.add_argument(
        "--endowment",
        type=float,
        default=DEFAULT_ENDOWMENT,
        help="Player 1 endowment",
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=DEFAULT_MULTIPLIER,
        help="Multiplier on amount sent to Player 2",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full interactions")
    args = parser.parse_args()

    if args.endowment <= 0:
        print("Error: --endowment must be positive")
        return

    if args.multiplier <= 0:
        print("Error: --multiplier must be positive")
        return

    receiver_sent_amounts = [
        amount for amount in DEFAULT_RECEIVER_SENT_AMOUNTS if amount <= args.endowment
    ]
    if args.endowment not in receiver_sent_amounts:
        receiver_sent_amounts.append(args.endowment)
        receiver_sent_amounts = sorted(set(receiver_sent_amounts))

    PRINT_INTERACTIONS = args.verbose

    print(f"Initializing model: {args.model}")
    try:
        llm = get_model_interface(args.model)
    except Exception as e:
        print(f"Error loading model {args.model}: {e}")
        return

    output_dir = os.path.join("data", "results", "trust_game", args.model.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)

    exp = TrustGameExperiment(
        endowment=args.endowment,
        multiplier=args.multiplier,
        receiver_sent_amounts=receiver_sent_amounts,
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
