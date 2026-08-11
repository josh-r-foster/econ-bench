"""Deterministic response parser fixtures for every active experiment."""

import importlib

import pytest


PARSER_CASES = [
    ("src.tasks.dictator", "parse_dollar_amount", "25%", (200,), 50.0),
    ("src.tasks.dictator", "parse_dollar_amount", "undecided", (100,), None),
    ("src.tasks.ultimatum", "parse_dollar_amount", "$12.50", (100,), 12.5),
    ("src.tasks.ultimatum", "parse_accept_reject", "ACCEPT", (), "ACCEPT"),
    ("src.tasks.ultimatum", "parse_accept_reject", "My answer is no", (), "REJECT"),
    ("src.tasks.ultimatum", "parse_accept_reject", "undecided", (), None),
    ("src.tasks.trust_game", "parse_dollar_amount", "I send $7.50", (10,), 7.5),
    ("src.tasks.trust_game", "parse_dollar_amount", "$11", (10,), None),
    ("src.tasks.stag_hunt", "parse_a_b", "Decision: B", (), "B"),
    ("src.tasks.stag_hunt", "parse_a_b", "undecided", (), None),
    ("src.tasks.beauty_contest", "parse_number", "My choice: 22", (0, 100), 22),
    ("src.tasks.beauty_contest", "parse_number", "101", (0, 100), None),
    ("src.tasks.centipede_game", "parse_pass_take", "Decision: TAKE", (), "TAKE"),
    ("src.tasks.centipede_game", "parse_pass_take", "undecided", (), None),
    ("src.tasks.public_goods", "parse_contribution", "I contribute $40", (100,), 40),
    ("src.tasks.public_goods", "parse_contribution", "$10.50", (100,), None),
    ("src.tasks.travellers_dilemma", "parse_number", "Claim: $90", (2, 100), 90),
    ("src.tasks.travellers_dilemma", "parse_number", "101", (2, 100), None),
    ("src.tasks.matching_pennies", "parse_heads_tails", "Answer: TAILS", (), "TAILS"),
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
