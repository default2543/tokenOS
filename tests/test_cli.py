import pytest

from tokenos.cli import _strategies, build_parser
from tokenos.config import ConfigurationError


def test_strategy_parser_rejects_duplicates_and_unknown_values() -> None:
    with pytest.raises(ConfigurationError, match="duplicates"):
        _strategies("no-memory,no-memory")
    with pytest.raises(ConfigurationError, match="drawn from"):
        _strategies("summary")


def test_strategy_parser_accepts_patchsearch() -> None:
    assert _strategies("full-history,patchsearch,patchsearch-explained") == (
        "full-history",
        "patchsearch",
        "patchsearch-explained",
    )


def test_resume_accepts_explicit_total_budget_cap() -> None:
    args = build_parser().parse_args(["resume", "run-1", "--budget-usd", "20"])
    assert args.run_id == "run-1"
    assert args.budget_usd == 20


def test_bigcodebench_benchmark_defaults_are_selected_later() -> None:
    args = build_parser().parse_args(
        ["benchmark", "--dataset", "bigcodebench-hard", "--all"]
    )
    assert args.dataset == "bigcodebench-hard"
    assert args.dataset_path is None
    assert args.evaluator_image is None


def test_bigcodebench_dataset_fetch_parser() -> None:
    args = build_parser().parse_args(
        ["dataset", "fetch", "--name", "bigcodebench-hard"]
    )
    assert args.name == "bigcodebench-hard"
    assert args.path is None


def test_preflight_parser_defaults_to_the_hard_diagnostic() -> None:
    args = build_parser().parse_args(["preflight"])
    assert args.dataset == "bigcodebench-hard"
    assert args.task_ids is None
    assert args.report_path is None


def test_preflight_selection_flags_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["preflight", "--all", "--task-ids", "B/1"])


def test_benchmark_can_request_preflight_filtering() -> None:
    args = build_parser().parse_args(
        ["benchmark", "--dataset", "bigcodebench-hard", "--all", "--skip-unviable-tasks"]
    )
    assert args.skip_unviable_tasks is True


def test_benchmark_does_not_filter_by_preflight_unless_asked() -> None:
    args = build_parser().parse_args(["benchmark", "--all"])
    assert args.skip_unviable_tasks is False
