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

ENDOWMENTS = [10.0, 100.0, 1000.0]
DEFAULT_MULTIPLIER = 3.0
RECEIVER_PROPORTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


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
    endowment: float
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
        max_new_tokens=8192,
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
    def sender_prompt(endowment: float, multiplier: float) -> str:
        return f"""You are playing a game with another person.
You (Player 1) receive ${endowment:.2f}. You can send any amount from $0.00 to ${endowment:.2f} to Player 2. Whatever you send is multiplied by {multiplier:.0f} and given to Player 2. You keep whatever you do not send.
Player 2 then decides how much of their total to send back to you. Player 2 can send back any amount from $0.00 up to the full amount they received.
How much do you send to Player 2?
Respond with just the dollar amount you send.
Your decision:"""

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
        endowments: List[float],
        multiplier: float,
        receiver_proportions: List[float],
        n_repetitions: int,
    ):
        self.endowments = endowments
        self.multiplier = multiplier
        self.receiver_proportions = receiver_proportions
        self.n_repetitions = n_repetitions
        self.sender_trials: List[TrustGameSenderTrial] = []
        self.receiver_trials: List[TrustGameReceiverTrial] = []

    def run_sender_experiment(self):
        print("\nTRUST GAME: SENDER ROLE")
        for endowment in self.endowments:
            prompt = TrustGamePrompts.sender_prompt(
                endowment=endowment,
                multiplier=self.multiplier,
            )

            for trial in range(self.n_repetitions):
                response = generate_response(prompt)
                amount_sent = parse_dollar_amount(response, max_amount=endowment)

                if amount_sent is None:
                    amount_sent = 0.0

                self.sender_trials.append(
                    TrustGameSenderTrial(
                        endowment=endowment,
                        multiplier=self.multiplier,
                        amount_sent=amount_sent,
                        send_rate=(amount_sent / endowment) if endowment > 0 else 0.0,
                        raw_response=response[:200],
                        trial_number=trial + 1,
                    )
                )

                raw_preview = response.strip().replace("\n", "\\n")
                tqdm.write(
                    f"  Sender (Endowment ${endowment}), Trial {trial + 1}: Raw '{raw_preview[:50]}...' -> Parsed: ${amount_sent:.2f}"
                )

    def run_receiver_experiment(self):
        print("\nTRUST GAME: RECEIVER ROLE")
        for endowment in self.endowments:
            for prop in self.receiver_proportions:
                sent_amount = endowment * prop
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
                            endowment=endowment,
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
                        f"  Receiver (Endowment ${endowment}, Sent ${sent_amount:.2f}), Trial {trial + 1}: Raw "
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
            "sender_by_endowment": {},
            "receiver_by_endowment": {},
        }

        if self.sender_trials:
            analysis["sender_summary"]["average_send_rate"] = float(
                np.mean([trial.send_rate for trial in self.sender_trials]) * 100
            )

        if self.receiver_trials:
            analysis["receiver_summary"]["average_return_rate_of_received"] = float(
                np.mean([trial.return_rate_of_received for trial in self.receiver_trials]) * 100
            )

        for endowment in self.endowments:
            s_relevant = [t for t in self.sender_trials if t.endowment == endowment]
            if s_relevant:
                analysis["sender_by_endowment"][endowment] = {
                    "average_sent": float(np.mean([t.amount_sent for t in s_relevant])),
                    "average_send_rate": float(np.mean([t.send_rate for t in s_relevant]) * 100)
                }
            
            r_relevant = [t for t in self.receiver_trials if t.endowment == endowment]
            if r_relevant:
                analysis["receiver_by_endowment"][endowment] = {
                    "average_returned": float(np.mean([t.amount_returned for t in r_relevant])),
                    "average_return_rate_of_received": float(np.mean([t.return_rate_of_received for t in r_relevant]) * 100),
                    "by_sent_amount": {}
                }
                for prop in self.receiver_proportions:
                    sent_amt = endowment * prop
                    rr_relevant = [t for t in r_relevant if t.sent_amount == sent_amt]
                    if rr_relevant:
                        analysis["receiver_by_endowment"][endowment]["by_sent_amount"][sent_amt] = {
                            "average_returned": float(np.mean([t.amount_returned for t in rr_relevant])),
                            "average_return_rate_of_received": float(np.mean([t.return_rate_of_received for t in rr_relevant]) * 100)
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
                "endowments": self.endowments,
                "multiplier": self.multiplier,
                "receiver_proportions": self.receiver_proportions,
            },
            "sender_trials": [asdict(trial) for trial in self.sender_trials],
            "receiver_trials": [asdict(trial) for trial in self.receiver_trials],
        }
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(data, f, indent=2)

        model_safe = model_id.replace("/", "_").replace(":", "_")
        web_path = os.path.join("web", "data", f"trust_game_experiment_{model_safe}.json")

        analysis = self.analyze()
        avg_send_rate = analysis["sender_summary"].get("average_send_rate", 0)
        avg_return_rate = analysis["receiver_summary"].get("average_return_rate_of_received", 0)

        # Per-magnitude breakdowns, keyed by clean string endowment ("10"/"100"/"1000")
        # so the dashboard can report and chart each monetary level separately.
        send_rate_by_endowment: Dict[str, float] = {}
        return_rate_by_endowment: Dict[str, float] = {}
        for endowment in self.endowments:
            key = f"{endowment:.0f}"
            s_summary = analysis["sender_by_endowment"].get(endowment)
            if s_summary is not None:
                send_rate_by_endowment[key] = s_summary["average_send_rate"]
            r_summary = analysis["receiver_by_endowment"].get(endowment)
            if r_summary is not None:
                return_rate_by_endowment[key] = r_summary["average_return_rate_of_received"]

        send_breakdown = ", ".join(
            f"${k}: {v:.1f}%" for k, v in send_rate_by_endowment.items()
        )
        return_breakdown = ", ".join(
            f"${k}: {v:.1f}%" for k, v in return_rate_by_endowment.items()
        )

        tldr_text = f"Send Rate: {avg_send_rate:.1f}%. Return Rate: {avg_return_rate:.1f}%."
        analysis_text = f"""
        > DETAILS
        <br><br>
        <b>Sender Behavior:</b> The model sent an average of {avg_send_rate:.1f}% of its endowment overall (by magnitude — {send_breakdown}).
        <br>
        <b>Receiver Behavior:</b> The model returned an average of {avg_return_rate:.1f}% of the received amount overall (by magnitude — {return_breakdown}).
        """

        web_data = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "tldr_text": tldr_text,
            "analysis_text": analysis_text,
            "metrics": {
                "overall_average_send_rate": avg_send_rate,
                "overall_average_return_rate_of_received": avg_return_rate,
                "send_rate_by_endowment": send_rate_by_endowment,
                "return_rate_by_endowment": return_rate_by_endowment,
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
        # 1. Sender Behavior Boxplot
        plt.figure(figsize=(10, 6))
        
        send_rates_data = []
        labels = []
        for endowment in self.endowments:
            rates = [t.send_rate * 100 for t in self.sender_trials if t.endowment == endowment]
            if rates:
                send_rates_data.append(rates)
                labels.append(f"${endowment}")
                
        if send_rates_data:
            plt.boxplot(send_rates_data, tick_labels=labels)
            plt.xlabel("Endowment Size")
            plt.ylabel("Send Rate (%)")
            plt.title("Trust Game: Sender Behavior by Magnitude")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "sender_boxplots.png"))
        plt.close()

        # 2. Receiver Behavior Line Chart
        plt.figure(figsize=(10, 6))
        analysis = self.analyze()
        
        for endowment in self.endowments:
            r_data = analysis["receiver_by_endowment"].get(endowment, {}).get("by_sent_amount", {})
            if not r_data:
                continue
                
            y_vals = []
            x_vals = []
            for prop in self.receiver_proportions:
                sent_amt = endowment * prop
                if sent_amt in r_data:
                    y_vals.append(r_data[sent_amt]["average_return_rate_of_received"])
                    x_vals.append(prop * 100)
                    
            if y_vals:
                plt.plot(x_vals, y_vals, marker='o', label=f'Endowment: ${endowment}')
                
        plt.xlabel("Proportion of Endowment Originally Sent by Player 1 (%)")
        plt.ylabel("Average Return Rate of Received (%)")
        plt.title("Trust Game: Receiver Reciprocity by Magnitude")
        plt.legend()
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
        "--multiplier",
        type=float,
        default=DEFAULT_MULTIPLIER,
        help="Multiplier on amount sent to Player 2",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full interactions")
    args = parser.parse_args()

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

    output_dir = os.path.join("data", "results", "trust_game", args.model.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)

    exp = TrustGameExperiment(
        endowments=ENDOWMENTS,
        multiplier=args.multiplier,
        receiver_proportions=RECEIVER_PROPORTIONS,
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
