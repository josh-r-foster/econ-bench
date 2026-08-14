"""Run opt-in live checks against configured model providers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_PROMPT = "Reply with exactly one word: Success"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an explicit live smoke check for one or more model endpoints"
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        dest="models",
        help="Provider model identifier. Repeat this option to test multiple models",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from dotenv import load_dotenv

    if args.env_file.is_file():
        load_dotenv(args.env_file)

    from src.models.registry import get_model_interface

    failures = 0
    for model_id in args.models:
        print(f"Testing {model_id}")
        try:
            interface = get_model_interface(model_id)
            response, _ = interface.generate_response(
                prompt=args.prompt,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
            if not response:
                raise RuntimeError("provider returned an empty response")
            print(f"PASS {model_id} {response.strip()[:200]}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {model_id} {type(exc).__name__} {exc}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
