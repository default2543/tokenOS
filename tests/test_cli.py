import pytest

from tokenos.cli import _strategies, build_parser, command_benchmark
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


def test_crosstask_strategy_requires_a_memory_store() -> None:
    args = build_parser().parse_args(
        ["benchmark", "--all", "--strategies", "patchsearch-crosstask"]
    )
    assert args.memory_store is None
    assert "patchsearch-crosstask" in _strategies(args.strategies)


def test_memory_store_flag_is_parsed() -> None:
    args = build_parser().parse_args(
        ["benchmark", "--all", "--strategies", "patchsearch-crosstask",
         "--memory-store", "20260831T133412Z-b22fb4"]
    )
    assert args.memory_store == "20260831T133412Z-b22fb4"


def test_benchmark_context_budget_defaults_to_unbounded() -> None:
    args = build_parser().parse_args(["benchmark", "--all"])
    assert args.context_budget is None


def test_benchmark_context_budget_is_parsed() -> None:
    args = build_parser().parse_args(
        ["benchmark", "--all", "--strategies", "patchsearch-with-code",
         "--context-budget", "1000"]
    )
    assert args.context_budget == 1000


@pytest.mark.parametrize(
    ("strategies", "budget", "message"),
    [
        ("full-history", "1000", "bounded-memory"),
        ("patchsearch-with-code", "0", "positive"),
    ],
)
def test_benchmark_rejects_a_context_budget_that_cannot_apply(
    tmp_path, strategies: str, budget: str, message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        command_benchmark(
            build_parser().parse_args(
                ["benchmark", "--dataset-path", str(tmp_path / "missing.jsonl.gz"),
                 "--task-ids", "HumanEval/0", "--strategies", strategies,
                 "--context-budget", budget]
            )
        )


def test_benchmark_output_cap_defaults_to_the_recorded_runs() -> None:
    args = build_parser().parse_args(["benchmark", "--all"])
    assert args.max_output_tokens == 4096
    raised = build_parser().parse_args(
        ["benchmark", "--all", "--max-output-tokens", "16384"]
    )
    assert raised.max_output_tokens == 16384


def test_best_code_is_a_valid_strategy() -> None:
    assert _strategies("full-history,patchsearch-best-code") == (
        "full-history", "patchsearch-best-code"
    )


def test_sliding_window_requires_a_context_budget(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="sliding-window needs"):
        command_benchmark(
            build_parser().parse_args(
                ["benchmark", "--dataset-path", str(tmp_path / "missing.gz"),
                 "--task-ids", "HumanEval/0", "--strategies", "sliding-window"]
            )
        )
