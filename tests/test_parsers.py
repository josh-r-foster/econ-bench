"""Deterministic response parser fixtures for every active experiment."""

import importlib

import pytest


PARSER_CASES = [
    ("src.tasks.dictator", "parse_dollar_amount", "TRANSFER=50", (200,), 50.0),
    ("src.tasks.dictator", "parse_dollar_amount", "undecided", (100,), None),
    ("src.tasks.ultimatum", "parse_dollar_amount", "OFFER=$12.50", (100,), 12.5),
    ("src.tasks.ultimatum", "parse_accept_reject", "DECISION=ACCEPT", (), "ACCEPT"),
    ("src.tasks.ultimatum", "parse_accept_reject", "My answer is no", (), None),
    ("src.tasks.ultimatum", "parse_accept_reject", "undecided", (), None),
    ("src.tasks.trust_game", "parse_dollar_amount", "SEND=7.50", (10, "send"), 7.5),
    ("src.tasks.trust_game", "parse_dollar_amount", "SEND=11", (10, "send"), None),
    ("src.tasks.stag_hunt", "parse_a_b", "CHOICE=B", (), "B"),
    ("src.tasks.stag_hunt", "parse_a_b", "undecided", (), None),
    ("src.tasks.beauty_contest", "parse_number", "CHOICE=22", (0, 100), 22),
    ("src.tasks.beauty_contest", "parse_number", "101", (0, 100), None),
    ("src.tasks.centipede_game", "parse_pass_take", "ACTION=TAKE", (), "TAKE"),
    ("src.tasks.centipede_game", "parse_pass_take", "undecided", (), None),
    ("src.tasks.public_goods", "parse_contribution", "CONTRIBUTION=40", (100,), 40),
    ("src.tasks.public_goods", "parse_contribution", "CONTRIBUTION=$10.50", (100,), 10.5),
    ("src.tasks.travellers_dilemma", "parse_number", "CLAIM=$90", (2, 100), 90),
    ("src.tasks.travellers_dilemma", "parse_number", "101", (2, 100), None),
    ("src.tasks.matching_pennies", "parse_heads_tails", "CHOICE=TAILS", (), "TAILS"),
    ("src.tasks.matching_pennies", "parse_heads_tails", "undecided", (), None),
]


@pytest.mark.parametrize(
    ("module_name", "function_name", "response", "args", "expected"),
    PARSER_CASES,
    ids=[f"{module}.{function}-{index}" for index, (module, function, *_rest) in enumerate(PARSER_CASES)],
)
def test_task_parser_fixture(module_name, function_name, response, args, expected):
    module = importlib.import_module(module_name)
    parser = getattr(module, function_name)
    actual = parser(response, *args)
    assert actual == expected, (
        f"{module_name}.{function_name} parsed {response!r} as {actual!r}, "
        f"expected {expected!r}"
    )


@pytest.mark.parametrize(
    ("module_name", "function_name", "response", "args", "expected"),
    [
        ("src.tasks.dictator", "parse_dollar_amount", "TRANSFER=5.25", (10,), 5.25),
        (
            "src.tasks.dictator", "parse_dollar_amount",
            "From the $10 pool, I give $5", (10,), None,
        ),
        ("src.tasks.ultimatum", "parse_dollar_amount", "OFFER=4.50", (10,), 4.5),
        ("src.tasks.ultimatum", "parse_accept_reject", "DECISION=REJECT", (), "REJECT"),
        ("src.tasks.trust_game", "parse_dollar_amount", "RETURN=7.25", (30, "return"), 7.25),
        ("src.tasks.stag_hunt", "parse_a_b", "CHOICE=B", (), "B"),
        (
            "src.tasks.stag_hunt", "parse_a_b",
            "As a cautious player, I choose B", (), None,
        ),
        ("src.tasks.beauty_contest", "parse_number", "CHOICE=22", (0, 100), 22),
        (
            "src.tasks.public_goods", "parse_contribution",
            "CONTRIBUTION=12.345", (100,), None,
        ),
        (
            "src.tasks.public_goods", "parse_contribution",
            "CONTRIBUTION=25%", (100,), None,
        ),
        (
            "src.tasks.travellers_dilemma", "parse_number",
            "CLAIM=7.30", (0.2, 10, 0.1), 7.3,
        ),
        (
            "src.tasks.travellers_dilemma", "parse_number",
            "CLAIM=7.35", (0.2, 10, 0.1), None,
        ),
    ],
)
def test_explicit_strategy_expression_parsers(
    module_name, function_name, response, args, expected
):
    parser = getattr(importlib.import_module(module_name), function_name)
    assert parser(response, *args) == expected


@pytest.mark.parametrize(
    ("parser", "response", "args"),
    [
        ("dictator", "I could give 20, but TRANSFER=0", (100,)),
        ("stag_hunt", "A seems safe, but CHOICE=B", ()),
        ("centipede_game", "PASS seems tempting, but ACTION=TAKE", ()),
        ("matching_pennies", "HEADS or perhaps CHOICE=TAILS", ()),
    ],
)
def test_contextual_prose_is_not_accepted_as_a_strategy(parser, response, args):
    module = importlib.import_module(f"src.tasks.{parser}")
    function = getattr(module, {
        "dictator": "parse_dollar_amount",
        "stag_hunt": "parse_a_b",
        "centipede_game": "parse_pass_take",
        "matching_pennies": "parse_heads_tails",
    }[parser])
    assert function(response, *args) is None


@pytest.mark.parametrize("module_name", ["src.tasks.independence", "src.tasks.time"])
@pytest.mark.parametrize("response, expected", [("A", "A"), ("B", "B"), ("?", None)])
def test_elicitation_parser_delegates_to_model_interface(
    monkeypatch, module_name, response, expected
):
    module = importlib.import_module(module_name)

    class FakeInterface:
        @staticmethod
        def parse_ab_choice(value):
            return value if value in {"A", "B"} else None

    monkeypatch.setattr(module, "llm", FakeInterface())
    assert module.parse_ab_choice(response) == expected
