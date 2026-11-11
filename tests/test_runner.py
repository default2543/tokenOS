from pathlib import Path
from threading import Lock

from tokenos.artifacts import EventStore
from tokenos.models import (
    AttemptRecord,
    EvaluationResult,
    FailureFeedback,
    ModelUsage,
    Problem,
    RunConfig,
)
from tokenos.provider import ModelResponse
from tokenos.runner import BenchmarkRunner
from tokenos.strategies import SYSTEM_INSTRUCTIONS, system_instructions


class FakeProvider:
    def __init__(self) -> None:
        self.prompts = []
        self.instructions = []
        self.lock = Lock()

    def generate(self, prompt: str, *, instructions: str | None = None) -> ModelResponse:
        with self.lock:
            self.prompts.append(prompt)
            self.instructions.append(instructions)
        value = "True" if (
            "PREVIOUS ATTEMPTS" in prompt or "FAILURE MEMORY" in prompt
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
    assert records[0].memories[0].content == "assert f(*()) == True"
    assert "assert f(*()) == True" in provider.prompts[1]
    assert summary["strategies"]["patchsearch"]["solved_tasks"] == 1
    assert summary["strategies"]["patchsearch"]["patches_generated"] == 1
    assert summary["strategies"]["patchsearch"]["patches_retrieved"] == 1
    assert summary["strategies"]["patchsearch"]["retry_model_calls"] == 1
    assert summary["strategies"]["patchsearch"]["retry_input_tokens"] == 20
    assert summary["strategies"]["patchsearch"]["patch_bearing_attempts"] == 1
    assert summary["strategies"]["patchsearch"]["patch_bearing_input_tokens"] == 20


def test_explained_patchsearch_is_persistent_and_measured(tmp_path: Path) -> None:
    problem = Problem("HumanEval/0", "def f():\n", "f")
    config = RunConfig(
        run_id="run",
        strategies=("patchsearch-explained",),
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
    assert records[0].memories[0].explanation == (
        "Previous candidate returned False but expected True."
    )
    assert (
        "Explanation: Previous candidate returned False but expected True."
        in provider.prompts[1]
    )
    assert "Constraint: assert f(*()) == True" in provider.prompts[1]
    metrics = summary["strategies"]["patchsearch-explained"]
    assert metrics["solved_tasks"] == 1
    assert metrics["patches_generated"] == 1
    assert metrics["patches_retrieved"] == 1
    assert metrics["explanations_generated"] == 1
    assert metrics["explanations_retrieved"] == 1
    assert metrics["explanation_bearing_attempts"] == 1
    assert metrics["explanation_bearing_input_tokens"] == 20
    assert metrics["explanation_retry_successes"] == 1


def test_provider_receives_instructions_matching_the_benchmark(tmp_path: Path) -> None:
    problem = Problem(
        "BigCodeBench/13", "def task_func():\n", "task_func", "bigcodebench-hard"
    )
    config = RunConfig(
        run_id="run",
        strategies=("no-memory",),
        task_ids=(problem.task_id,),
        dataset_path="data.gz",
        dataset_name="bigcodebench-hard",
        max_attempts=1,
        budget_usd=10,
        input_usd_per_mtok=1,
        output_usd_per_mtok=1,
    )
    store = EventStore(tmp_path, config.run_id, create=True)
    store.write_manifest(config)
    provider = FakeProvider()
    BenchmarkRunner(config, provider, FakeEvaluator(), store).run([problem])
    assert provider.instructions == [system_instructions("bigcodebench-hard")]
    assert provider.instructions != [SYSTEM_INSTRUCTIONS]


def test_humaneval_keeps_the_strict_instructions(tmp_path: Path) -> None:
    problem = Problem("HumanEval/0", "def f():\n", "f")
    config = RunConfig(
        run_id="run",
        strategies=("no-memory",),
        task_ids=(problem.task_id,),
        dataset_path="data.gz",
        max_attempts=1,
        budget_usd=10,
        input_usd_per_mtok=1,
        output_usd_per_mtok=1,
    )
    store = EventStore(tmp_path, config.run_id, create=True)
    store.write_manifest(config)
    provider = FakeProvider()
    BenchmarkRunner(config, provider, FakeEvaluator(), store).run([problem])
    assert provider.instructions == [SYSTEM_INSTRUCTIONS]


def _store_with(task_id: str, content: str):
    from tokenos.models import FailureMemory
    from tokenos.store import MemoryStore
    other = Problem(task_id, "def f(archive):\n  archive files\n", "f")
    record = AttemptRecord(
        run_id="r", task_id=task_id, strategy="patchsearch", attempt=1, prompt="p",
        completion="c", solution="s", response_id=None, request_id=None,
        usage=ModelUsage(), evaluation=EvaluationResult("fail", "fail", False),
        model_latency_seconds=0, evaluation_latency_seconds=0,
        total_latency_seconds=0, estimated_cost_usd=0, created_at="now",
        memories=(FailureMemory(task_id=task_id, kind="failure_summary",
            content=content, source_attempt=1, failure_kind="runtime_error",
            fingerprint="fp"),),
    )
    return MemoryStore.from_records([record], {task_id: other})


def _crosstask_config(strategy, task_id):
    return RunConfig(
        run_id="run", strategies=(strategy,), task_ids=(task_id,),
        dataset_path="data.gz", max_attempts=1, budget_usd=10,
        input_usd_per_mtok=1, output_usd_per_mtok=1,
    )


def test_crosstask_strategy_receives_memory_from_other_tasks(tmp_path: Path) -> None:
    problem = Problem("HumanEval/0", "def f(archive):\n  archive files\n", "f")
    config = _crosstask_config("patchsearch-crosstask", problem.task_id)
    store = EventStore(tmp_path, config.run_id, create=True)
    store.write_manifest(config)
    provider = FakeProvider()
    BenchmarkRunner(
        config, provider, FakeEvaluator(), store,
        memory_store=_store_with("HumanEval/9", "RUNTIME ERROR: archive missing."),
    ).run([problem])
    assert "RELATED FAILURES FROM OTHER TASKS" in provider.prompts[0]
    assert "HumanEval/9" in provider.prompts[0]


def test_plain_patchsearch_ignores_the_cross_task_store(tmp_path: Path) -> None:
    # The store must not leak into the strategy it is being compared against.
    problem = Problem("HumanEval/0", "def f(archive):\n  archive files\n", "f")
    config = _crosstask_config("patchsearch", problem.task_id)
    store = EventStore(tmp_path, config.run_id, create=True)
    store.write_manifest(config)
    provider = FakeProvider()
    BenchmarkRunner(
        config, provider, FakeEvaluator(), store,
        memory_store=_store_with("HumanEval/9", "RUNTIME ERROR: archive missing."),
    ).run([problem])
    assert "RELATED FAILURES" not in provider.prompts[0]


def test_evaluator_value_error_is_infrastructure_not_invalid_output(
    tmp_path: Path,
) -> None:
    # AdapterError subclasses ValueError; it once masqueraded as the model's
    # malformed output and was scored as a failed attempt.
    class BrokenEvaluator:
        def evaluate(self, problem: Problem, solution: str) -> EvaluationResult:
            raise ValueError("unknown benchmark")

    problem = Problem("HumanEval/0", "def f():\n", "f")
    config = RunConfig(
        run_id="run",
        strategies=("no-memory",),
        task_ids=(problem.task_id,),
        dataset_path="data.gz",
        max_attempts=1,
        input_usd_per_mtok=1,
        cached_input_usd_per_mtok=0.1,
        output_usd_per_mtok=1,
    )
    store = EventStore(tmp_path, config.run_id, create=True)
    store.write_manifest(config)
    summary = BenchmarkRunner(
        config, FakeProvider(), BrokenEvaluator(), store
    ).run([problem])
    assert summary["stop_reason"] == "evaluator_error"
    assert not store.completed_attempts()


def _with_code_run(tmp_path: Path, context_budget: int | None, run_id: str):
    problem = Problem("HumanEval/0", "def f():\n", "f")
    config = RunConfig(
        run_id=run_id,
        strategies=("patchsearch-with-code",),
        task_ids=(problem.task_id,),
        dataset_path="data.gz",
        max_attempts=2,
        concurrency=1,
        input_usd_per_mtok=1,
        cached_input_usd_per_mtok=0.1,
        output_usd_per_mtok=1,
        context_budget=context_budget,
    )
    store = EventStore(tmp_path, config.run_id, create=True)
    store.write_manifest(config)
    provider = FakeProvider()
    summary = BenchmarkRunner(config, provider, FakeEvaluator(), store).run([problem])
    contexts = [
        event["context"]
        for event in store.events()
        if event["type"] == "request_started"
    ]
    return summary, provider, contexts


def test_unbounded_run_logs_the_compiled_context_without_trimming(
    tmp_path: Path,
) -> None:
    _, provider, contexts = _with_code_run(tmp_path, None, "unbounded")
    assert "MOST RECENT ATTEMPT (attempt 1)" in provider.prompts[1]
    assert contexts[1]["included_code"] is True
    assert contexts[1]["budget_tokens"] is None
    assert not contexts[1]["dropped_code"]


def test_binding_context_budget_drops_the_code_before_memory(tmp_path: Path) -> None:
    _, _, unbounded = _with_code_run(tmp_path, None, "unbounded")
    budget = unbounded[1]["estimated_tokens"] - 1
    _, provider, contexts = _with_code_run(tmp_path, budget, "bounded")
    assert "MOST RECENT ATTEMPT" not in provider.prompts[1]
    assert "assert f(*()) == True" in provider.prompts[1]
    assert contexts[1]["dropped_code"] is True
    assert contexts[1]["memories"] == 1
    assert contexts[1]["estimated_tokens"] <= budget


def test_task_over_the_context_budget_stops_without_a_model_call(
    tmp_path: Path,
) -> None:
    summary, provider, _ = _with_code_run(tmp_path, 1, "tiny")
    assert summary["stop_reason"] == "context_budget_exceeded"
    assert provider.prompts == []


def test_full_history_is_never_compiled(tmp_path: Path) -> None:
    problem = Problem("HumanEval/0", "def f():\n", "f")
    config = RunConfig(
        run_id="run",
        strategies=("full-history",),
        task_ids=(problem.task_id,),
        dataset_path="data.gz",
        max_attempts=2,
        input_usd_per_mtok=1,
        cached_input_usd_per_mtok=0.1,
        output_usd_per_mtok=1,
        context_budget=1,
    )
    store = EventStore(tmp_path, config.run_id, create=True)
    store.write_manifest(config)
    provider = FakeProvider()
    summary = BenchmarkRunner(config, provider, FakeEvaluator(), store).run([problem])
    assert summary["strategies"]["full-history"]["solved_tasks"] == 1
    assert "PREVIOUS ATTEMPTS" in provider.prompts[1]


class AlwaysWrongProvider(FakeProvider):
    def generate(self, prompt: str, *, instructions: str | None = None) -> ModelResponse:
        self.prompts.append(prompt)
        return ModelResponse(
            output_text='{"completion":"    return False"}',
            response_id="resp",
            request_id="req",
            usage=ModelUsage(input_tokens=20, output_tokens=10),
            latency_seconds=0.01,
        )


def _stuck_run(tmp_path: Path, strategy: str, stop: int | None, run_id: str = "run"):
    problem = Problem("HumanEval/0", "def f():\n", "f")
    config = RunConfig(
        run_id=run_id,
        strategies=(strategy,),
        task_ids=(problem.task_id,),
        dataset_path="data.gz",
        max_attempts=5,
        concurrency=1,
        input_usd_per_mtok=1,
        cached_input_usd_per_mtok=0.1,
        output_usd_per_mtok=1,
        stop_after_identical=stop,
    )
    store = EventStore(tmp_path, config.run_id, create=True)
    store.write_manifest(config)
    provider = AlwaysWrongProvider()
    BenchmarkRunner(config, provider, FakeEvaluator(), store).run([problem])
    return store, provider


def test_trajectory_is_abandoned_after_identical_failures(tmp_path: Path) -> None:
    store, provider = _stuck_run(tmp_path, "patchsearch-adaptive", stop=2)
    assert len(provider.prompts) == 2
    abandoned = [e for e in store.events() if e["type"] == "trajectory_abandoned"]
    assert abandoned and abandoned[0]["identical_failures"] == 2


def test_without_a_stop_limit_every_attempt_runs(tmp_path: Path) -> None:
    _, provider = _stuck_run(tmp_path, "patchsearch-adaptive", stop=None)
    assert len(provider.prompts) == 5


def test_full_history_is_never_abandoned(tmp_path: Path) -> None:
    # It stores no memories, so it has no failure fingerprints to compare.
    _, provider = _stuck_run(tmp_path, "full-history", stop=2)
    assert len(provider.prompts) == 5


def test_stop_simulation_matches_a_live_stop(tmp_path: Path) -> None:
    from tokenos.replay import stop_simulation

    store, _ = _stuck_run(tmp_path, "patchsearch-adaptive", stop=None, run_id="full")
    never, at_two = stop_simulation(store.completed_attempts(), [2])[
        "patchsearch-adaptive"
    ]
    assert (never["attempts"], never["abandoned"]) == (5, 0)
    assert (at_two["attempts"], at_two["abandoned"], at_two["solved"]) == (2, 1, 0)
    assert at_two["output_tokens"] == 2 * 10

    live, _ = _stuck_run(tmp_path, "patchsearch-adaptive", stop=2, run_id="live")
    assert len(live.completed_attempts()) == at_two["attempts"]


def test_export_workload_keeps_exact_prompts_in_trajectory_order(tmp_path: Path) -> None:
    from tokenos.replay import export_workload

    store, provider = _stuck_run(tmp_path, "patchsearch-adaptive", stop=None)
    rows = export_workload(store.completed_attempts(), "be terse")
    assert [row["attempt"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["prompt"] for row in rows] == provider.prompts
    assert rows[0]["instructions"] == "be terse"
    assert rows[0]["recorded_output_tokens"] == 10


def test_retry_provider_serves_only_attempts_after_the_first(tmp_path: Path) -> None:
    problem = Problem("HumanEval/0", "def f():\n", "f")
    config = RunConfig(
        run_id="run", strategies=("patchsearch-adaptive",),
        task_ids=(problem.task_id,), dataset_path="data.gz", max_attempts=3,
        concurrency=1, input_usd_per_mtok=1, cached_input_usd_per_mtok=0.1,
        output_usd_per_mtok=1, retry_reasoning_effort="medium",
    )
    store = EventStore(tmp_path, config.run_id, create=True)
    store.write_manifest(config)
    first, retries = AlwaysWrongProvider(), AlwaysWrongProvider()
    BenchmarkRunner(config, first, FakeEvaluator(), store,
                    retry_provider=retries).run([problem])
    assert len(first.prompts) == 1
    assert len(retries.prompts) == 2
