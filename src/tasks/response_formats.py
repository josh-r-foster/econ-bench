"""Shared parsers for the explicit strategy expressions used in prompts."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Iterable


_NUMBER = r"(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?"


def _decimal(token: str) -> Decimal | None:
    try:
        return Decimal(token.replace(",", ""))
    except InvalidOperation:
        return None


def parse_bounded_amount(
    response: str,
    *,
    maximum: float,
    labels: Iterable[str],
    minimum: float = 0,
    increment: float | None = None,
    allow_percentage: bool = False,
    max_decimal_places: int | None = 2,
) -> float | None:
    """Parse one complete labeled amount expression and enforce its feasible set."""
    cleaned = response.strip()
    label_pattern = "|".join(re.escape(label) for label in labels)
    labeled = re.fullmatch(
        rf"(?i)(?:{label_pattern})\s*=\s*\$?\s*({_NUMBER})\s*(%)?",
        cleaned,
    )
    if labeled is None:
        return None

    lower = Decimal(str(minimum))
    upper = Decimal(str(maximum))
    step = Decimal(str(increment)) if increment is not None else None
    token = labeled.group(1)
    if max_decimal_places is not None:
        fractional = token.partition(".")[2]
        if len(fractional) > max_decimal_places:
            return None
    value = _decimal(token)
    if value is None:
        return None
    if bool(labeled.group(2)):
        if not allow_percentage or not Decimal("0") <= value <= Decimal("100"):
            return None
        value = upper * value / Decimal("100")
    if not lower <= value <= upper:
        return None
    if step is not None and (value - lower) % step != 0:
        return None
    return float(value)


def parse_labeled_choice(
    response: str,
    *,
    choices: Iterable[str],
    labels: Iterable[str] = ("choice",),
) -> str | None:
    """Parse one complete labeled categorical expression."""
    allowed = {choice.upper() for choice in choices}
    cleaned = response.strip()
    label_pattern = "|".join(re.escape(label) for label in labels)
    choice_pattern = "|".join(
        re.escape(choice) for choice in sorted(allowed, key=len, reverse=True)
    )
    match = re.fullmatch(
        rf"(?i)(?:{label_pattern})\s*=\s*({choice_pattern})",
        cleaned,
    )
    return match.group(1).upper() if match else None
