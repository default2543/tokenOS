import pytest

from tokenos.cli import _strategies, build_parser
from tokenos.config import ConfigurationError


def test_strategy_parser_rejects_duplicates_and_unknown_values() -> None:
    with pytest.raises(ConfigurationError, match="duplicates"):
        _strategies("no-memory,no-memory")
    with pytest.raises(ConfigurationError, match="drawn from"):
        _strategies("summary")


def test_resume_accepts_explicit_total_budget_cap() -> None:
    args = build_parser().parse_args(["resume", "run-1", "--budget-usd", "20"])
    assert args.run_id == "run-1"
    assert args.budget_usd == 20
