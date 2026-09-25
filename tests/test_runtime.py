from dataclasses import replace
import gzip
import json
from pathlib import Path

import pytest

from tokenos.artifacts import EventStore
from tokenos.cli import main
from tokenos.compiler import (
    ContextBudgetError,
    compile_context,
    estimate_tokens,
)
from tokenos.config import ConfigurationError
from tokenos.models import (
    AttemptRecord,
    EvaluationResult,
    FailureEvidence,
    FailureMemory,
    ModelUsage,
    Problem,
    RunConfig,
)
from tokenos.provider import ModelResponse
from tokenos.replay import budget_sweep, context_growth
from tokenos.runtime import Runtime
from tokenos.strategies import build_prompt

PROBLEM = Problem("HumanEval/0", 'def f(x):\n    """Return x doubled."""\n', "f")


def _memory(attempt: int, content: str, kind: str = "failure_summary") -> FailureMemory:
    return FailureMemory(
        task_id=PROBLEM.task_id,
        kind=kind,  # type: ignore[arg-type]
        content=content,
        source_attempt=attempt,
        failure_kind="wrong_answer",
        fingerprint=f"{kind}-{attempt}-{content}",
    )


def _record(attempt: int, completion: str, memories=(), strategy="patchsearch-with-code"):
    return AttemptRecord(
        run_id="run",
        task_id=PROBLEM.task_id,
        strategy=strategy,
        attempt=attempt,
        prompt="",
        completion=completion,
        solution=PROBLEM.prompt + completion,
        response_id=None,
        request_id=None,
        usage=ModelUsage(input_tokens=100 * attempt, output_tokens=10),
        evaluation=EvaluationResult("fail", "fail", False),
        model_latency_seconds=0.0,
        evaluation_latency_seconds=0.0,
        total_latency_seconds=0.0,
        estimated_cost_usd=0.0,
        created_at="",
        memories=tuple(memories),
    )


def test_estimate_is_conservative_for_ascii_and_multibyte_text() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("abcd") == 2
    assert estimate_tokens("é") == 1


def test_unbounded_compile_reproduces_the_benchmarked_prompt() -> None:
    memory = _memory(1, "f(2) returned 3, expected 4")
    history = [_record(1, "    return x + 1\n", [memory])]
    context = compile_context(
        "patchsearch-with-code", PROBLEM, 2, history, memories=[memory]
    )
    assert context.prompt == build_prompt(
        "patchsearch-with-code", PROBLEM, 2, history, memories=[memory]
    )
    assert context.included_code and not context.trimmed
    assert context.budget_tokens is None


def test_budget_drops_code_before_task_local_memory() -> None:
    memory = _memory(1, "f(2) returned 3, expected 4")
    history = [_record(1, "    return x + 1\n" + "    # padding\n" * 200, [memory])]
    full = compile_context(
        "patchsearch-with-code", PROBLEM, 2, history, memories=[memory]
    )
    no_code = compile_context(
        "patchsearch", PROBLEM, 2, history, memories=[memory]
    )
    context = compile_context(
        "patchsearch-with-code",
        PROBLEM,
        2,
        history,
        memories=[memory],
        budget_tokens=no_code.estimated_tokens + 60,
    )
    assert full.estimated_tokens > context.budget_tokens
    assert context.dropped_code and not context.included_code
    assert context.memories == (memory,)
    assert context.estimated_tokens <= context.budget_tokens
    assert "MOST RECENT ATTEMPT" not in context.prompt


def test_budget_keeps_constraints_over_summaries_and_newer_summaries_first() -> None:
    constraint = _memory(1, "assert f(2) == 4", kind="executable_constraint")
    older = _memory(2, "older summary " + "x" * 300)
    newer = _memory(3, "newer summary " + "y" * 300)
    memories = [constraint, older, newer]
    only_two = compile_context(
        "patchsearch", PROBLEM, 4, [], memories=[constraint, newer]
    )
    context = compile_context(
        "patchsearch",
        PROBLEM,
        4,
        [],
        memories=memories,
        budget_tokens=only_two.estimated_tokens,
    )
    assert context.memories == (constraint, newer)
    assert context.dropped_memories == (older,)


def test_transferred_memory_only_spends_what_task_memory_leaves() -> None:
    local = _memory(1, "local failure")
    remote = FailureMemory(
        task_id="HumanEval/9",
        kind="failure_summary",
        content="unrelated " * 100,
        source_attempt=1,
        failure_kind="runtime_error",
        fingerprint="remote",
    )
    required = compile_context("patchsearch", PROBLEM, 2, [], memories=[local])
    context = compile_context(
        "patchsearch",
        PROBLEM,
        2,
        [],
        memories=[local],
        transferred=[remote],
        budget_tokens=required.estimated_tokens + 5,
    )
    assert context.memories == (local,)
    assert context.dropped_transferred == (remote,)


def test_task_that_cannot_fit_is_refused_rather_than_truncated() -> None:
    with pytest.raises(ContextBudgetError, match="before any memory"):
        compile_context("patchsearch", PROBLEM, 1, [], budget_tokens=10)


class ScriptedProvider:
    def __init__(self, completions: list[str]) -> None:
        self.completions = list(completions)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, instructions: str | None = None) -> ModelResponse:
        self.prompts.append(prompt)
        return ModelResponse(
            output_text=json.dumps({"completion": self.completions.pop(0)}),
            response_id="resp",
            request_id="req",
            usage=ModelUsage(input_tokens=len(prompt) // 4, output_tokens=5),
            latency_seconds=0.0,
        )


class DoublingEvaluator:
    """Passes only a correct doubling, with oracle evidence otherwise."""

    def evaluate(self, problem: Problem, solution: str) -> EvaluationResult:
        namespace: dict = {}
        try:
            exec(solution, namespace)
            actual = namespace["f"](2)
        except Exception as exc:
            return EvaluationResult(
                "fail",
                "fail",
                False,
                FailureEvidence(kind="runtime_error", message=type(exc).__name__),
            )
        if actual == 4:
            return EvaluationResult("pass", "pass", True)
        return EvaluationResult(
            "fail",
            "fail",
            False,
            FailureEvidence(
                kind="wrong_answer",
                message="mismatch",
                input_repr="(2,)",
                expected_repr="4",
                actual_repr=repr(actual),
            ),
        )


def test_runtime_learns_from_a_failure_and_solves_on_retry() -> None:
    provider = ScriptedProvider(["    return x + 1\n", "    return x * 2\n"])
    runtime = Runtime(4000, provider=provider, evaluator=DoublingEvaluator())
    result = runtime.run(PROBLEM)

    assert result.passed
    assert [item.record.attempt for item in result.attempts] == [1, 2]
    assert result.stop_reason is None
    assert result.solution.endswith("return x * 2\n")
    retry_prompt = provider.prompts[1]
    assert "MOST RECENT ATTEMPT (attempt 1)" in retry_prompt
    assert "return x + 1" in retry_prompt
    assert "FAILURE MEMORY" in retry_prompt
    assert result.attempts[1].context.memories
    assert result.total_tokens == result.input_tokens + result.output_tokens
    assert result.to_dict()["attempts"][0]["failure_kind"] == "wrong_answer"


def test_runtime_scores_malformed_output_as_a_failed_attempt() -> None:
    class Malformed(ScriptedProvider):
        def generate(self, prompt, *, instructions=None):
            response = super().generate(prompt, instructions=instructions)
            return ModelResponse("not json", None, None, response.usage, 0.0)

    runtime = Runtime(
        4000,
        provider=Malformed(["", ""]),
        evaluator=DoublingEvaluator(),
        max_attempts=2,
    )
    result = runtime.run(PROBLEM)
    assert not result.passed
    assert result.stop_reason == "max_attempts"
    assert {
        item.record.evaluation.feedback.kind for item in result.attempts
    } == {"invalid_output"}


def test_runtime_stops_at_its_spend_cap() -> None:
    config = RunConfig(
        run_id="run",
        strategies=("patchsearch-with-code",),
        task_ids=(),
        dataset_path="",
        budget_usd=1e-9,
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=1.0,
    )
    provider = ScriptedProvider(["    return x * 2\n"])
    runtime = Runtime(
        4000, provider=provider, evaluator=DoublingEvaluator(), config=config
    )
    result = runtime.run(PROBLEM)
    assert result.stop_reason == "budget_exhausted"
    assert not result.attempts and not provider.prompts


def test_runtime_rejects_baselines_and_bad_budgets() -> None:
    evaluator = DoublingEvaluator()
    provider = ScriptedProvider([])
    with pytest.raises(ConfigurationError, match="baseline"):
        Runtime(provider=provider, evaluator=evaluator, strategy="full-history")
    with pytest.raises(ConfigurationError, match="positive"):
        Runtime(0, provider=provider, evaluator=evaluator)


def test_runtime_without_a_provider_requires_azure_configuration() -> None:
    with pytest.raises(ConfigurationError, match="AZURE_OPENAI_ENDPOINT"):
        Runtime(env={})


def test_growth_and_sweep_read_recorded_runs() -> None:
    memory = _memory(1, "f(2) returned 3, expected 4")
    first = _record(1, "    return x + 1\n", [memory])
    retry_prompt = build_prompt(
        "patchsearch-with-code", PROBLEM, 2, [first], memories=[memory]
    )
    second = _replace_prompt(_record(2, "    return x\n"), retry_prompt)
    first = _replace_prompt(
        first, build_prompt("patchsearch-with-code", PROBLEM, 1, [], memories=[])
    )

    growth = context_growth([first, second])
    assert [row["mean_input_tokens"] for row in growth["patchsearch-with-code"]] == [
        100,
        200,
    ]

    sweep = budget_sweep([first, second], {PROBLEM.task_id: PROBLEM}, [None, 10**6])
    unbounded, generous = sweep["patchsearch-with-code"]
    assert unbounded["reproduced"] == 2
    assert unbounded["attempts"] == 2
    assert generous["trimmed"] == 0
    assert generous["estimated_tokens"] == unbounded["estimated_tokens"]


def _replace_prompt(record: AttemptRecord, prompt: str) -> AttemptRecord:
    return replace(record, prompt=prompt)


def test_growth_and_sweep_commands(tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "data.jsonl.gz"
    with gzip.open(dataset, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "task_id": PROBLEM.task_id,
                    "prompt": PROBLEM.prompt,
                    "entry_point": "f",
                    "canonical_solution": "    return x * 2\n",
                    "test": "",
                }
            )
            + "\n"
        )
    config = RunConfig(
        run_id="run",
        strategies=("patchsearch-with-code",),
        task_ids=(PROBLEM.task_id,),
        dataset_path=str(dataset),
    )
    store = EventStore(tmp_path, "run", create=True)
    store.write_manifest(config)
    record = _replace_prompt(
        _record(1, "    return x\n"),
        build_prompt("patchsearch-with-code", PROBLEM, 1, [], memories=[]),
    )
    store.append("attempt_completed", record=record.to_dict())

    assert main(["growth", "run", "--runs-root", str(tmp_path)]) == 0
    assert "patchsearch-with-code" in capsys.readouterr().out
    assert main(["sweep", "run", "--runs-root", str(tmp_path), "--budgets", "4000"]) == 0
    out = capsys.readouterr().out
    assert "1 prompts reproduced byte-for-byte" in out
