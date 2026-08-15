from pathlib import Path
from threading import Lock

from tokenos.artifacts import EventStore
from tokenos.models import (
    EvaluationResult,
    FailureFeedback,
    ModelUsage,
    Problem,
    RunConfig,
)
from tokenos.provider import ModelResponse
from tokenos.runner import BenchmarkRunner


class FakeProvider:
    def __init__(self) -> None:
        self.prompts = []
        self.lock = Lock()

    def generate(self, prompt: str) -> ModelResponse:
        with self.lock:
            self.prompts.append(prompt)
        value = "True" if (
            "PREVIOUS ATTEMPTS" in prompt or "EXECUTABLE MEMORY" in prompt
        ) else "False"
        return ModelResponse(
            output_text='{"completion":"    return ' + value + '"}',
            response_id="resp",
            request_id="req",
            usage=ModelUsage(input_tokens=20, output_tokens=10),
            latency_seconds=0.01,
        )


class FakeEvaluator:
    def evaluate(self, problem: Problem, solution: str) -> EvaluationResult:
        passed = "return True" in solution
        return EvaluationResult(
            base_status="pass" if passed else "fail",
            plus_status="pass" if passed else "fail",
            passed=passed,
            feedback=None
            if passed
            else FailureFeedback(
                kind="wrong_answer",
                message="mismatch",
                input_repr="()",
                expected_repr="True",
                actual_repr="False",
            ),
        )


def test_end_to_end_strategies_and_artifacts(tmp_path: Path) -> None:
    problem = Problem("HumanEval/0", "def f():\n", "f")
    config = RunConfig(
        run_id="run",
        strategies=("no-memory", "full-history"),
        task_ids=(problem.task_id,),
        dataset_path="data.gz",
        max_attempts=2,
        concurrency=2,
        budget_usd=10,
        input_usd_per_mtok=1,
        cached_input_usd_per_mtok=0.1,
        output_usd_per_mtok=1,
    )
    store = EventStore(tmp_path, config.run_id, create=True)
    store.write_manifest(config)
    provider = FakeProvider()
    summary = BenchmarkRunner(
        config, provider, FakeEvaluator(), store
    ).run([problem])

    assert summary["strategies"]["no-memory"]["solved_tasks"] == 0
    assert summary["strategies"]["full-history"]["solved_tasks"] == 1
    assert summary["strategies"]["full-history"]["solve@3"] == 1.0
    no_memory_prompts = [p for p in provider.prompts if "unavailable" in p]
    assert all("Expected: True" not in prompt for prompt in no_memory_prompts)
    assert store.summary_path.exists()
    assert store.samples_path.exists()
    assert len(store.completed_attempts()) == 4


def test_budget_exhaustion_produces_partial_resumable_run(tmp_path: Path) -> None:
    problem = Problem("HumanEval/0", "def f():\n", "f")
    config = RunConfig(
        run_id="run",
        strategies=("no-memory",),
        task_ids=(problem.task_id,),
        dataset_path="data.gz",
        budget_usd=0.000001,
        input_usd_per_mtok=1,
        cached_input_usd_per_mtok=0.1,
        output_usd_per_mtok=10,
    )
    store = EventStore(tmp_path, config.run_id, create=True)
    store.write_manifest(config)
    summary = BenchmarkRunner(
        config, FakeProvider(), FakeEvaluator(), store
    ).run([problem])
    assert summary["complete"] is False
    assert summary["stop_reason"] == "budget_exhausted"
    assert any(event["type"] == "budget_exhausted" for event in store.events())


def test_orphaned_started_request_is_repeated_on_resume(tmp_path: Path) -> None:
    problem = Problem("HumanEval/0", "def f():\n", "f")
    config = RunConfig(
        run_id="run",
        strategies=("full-history",),
        task_ids=(problem.task_id,),
        dataset_path="data.gz",
        max_attempts=2,
        concurrency=1,
        budget_usd=10,
        input_usd_per_mtok=1,
        cached_input_usd_per_mtok=0.1,
        output_usd_per_mtok=1,
    )
    store = EventStore(tmp_path, config.run_id, create=True)
    store.write_manifest(config)
    store.append(
        "request_started",
        task_id=problem.task_id,
        strategy="full-history",
        attempt=1,
        prompt="orphan",
    )
    provider = FakeProvider()
    BenchmarkRunner(config, provider, FakeEvaluator(), store).run([problem])
    assert [record.attempt for record in store.completed_attempts()] == [1, 2]
    assert len(provider.prompts) == 2


def test_patchsearch_stores_and_retrieves_counterexample(tmp_path: Path) -> None:
    problem = Problem("HumanEval/0", "def f():\n", "f")
    config = RunConfig(
        run_id="run",
        strategies=("patchsearch",),
        task_ids=(problem.task_id,),
        dataset_path="data.gz",
        max_attempts=2,
        concurrency=1,
        budget_usd=10,
        input_usd_per_mtok=1,
        cached_input_usd_per_mtok=0.1,
        output_usd_per_mtok=1,
    )
    store = EventStore(tmp_path, config.run_id, create=True)
    store.write_manifest(config)
    provider = FakeProvider()
    summary = BenchmarkRunner(config, provider, FakeEvaluator(), store).run([problem])
    records = store.completed_attempts()
    assert records[0].patches[0].assertion == "assert f(*()) == True"
    assert "assert f(*()) == True" in provider.prompts[1]
    assert summary["strategies"]["patchsearch"]["solved_tasks"] == 1
    assert summary["strategies"]["patchsearch"]["patches_generated"] == 1
    assert summary["strategies"]["patchsearch"]["patches_retrieved"] == 1
    assert summary["strategies"]["patchsearch"]["retry_model_calls"] == 1
    assert summary["strategies"]["patchsearch"]["retry_input_tokens"] == 20
    assert summary["strategies"]["patchsearch"]["patch_bearing_attempts"] == 1
    assert summary["strategies"]["patchsearch"]["patch_bearing_input_tokens"] == 20
