from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
import threading
import time
from typing import Protocol, Sequence

from tokenos.artifacts import EventStore
from tokenos.budget import BudgetManager
from tokenos.config import utc_now
from tokenos.metrics import summarize
from tokenos.models import (
    AttemptRecord,
    EvaluationResult,
    FailureFeedback,
    ModelUsage,
    Problem,
    RunConfig,
    StrategyName,
)
from tokenos.provider import ModelProviderError, ModelResponse
from tokenos.patchsearch.extractor import extract_memory
from tokenos.patchsearch.retriever import retrieve_with_store
from tokenos.store import MemoryStore
from tokenos.strategies import (
    PATCHSEARCH_STRATEGIES,
    build_prompt,
    parse_completion,
    system_instructions,
)


class Provider(Protocol):
    def generate(
        self, prompt: str, *, instructions: str | None = None
    ) -> ModelResponse: ...


class Evaluator(Protocol):
    def evaluate(self, problem: Problem, solution: str) -> EvaluationResult: ...


def score_output(
    problem: Problem, output_text: str, evaluator: Evaluator
) -> tuple[str, str, EvaluationResult, float]:
    """Parse a model response and evaluate it.

    Returns the completion, the full solution, the evaluation, and evaluation
    latency. A malformed response is a failed attempt with `invalid_output`
    evidence rather than an error, because it is the model's mistake; evaluator
    failures propagate, because they are infrastructure's.
    """
    try:
        completion = parse_completion(output_text)
    except ValueError as exc:
        return output_text, problem.prompt, EvaluationResult(
            base_status="fail",
            plus_status="fail",
            passed=False,
            feedback=FailureFeedback(
                kind="invalid_output",
                message=str(exc),
                benchmark=problem.benchmark,
                provenance="tokenos-response-parser",
            ),
        ), 0.0
    solution = problem.prompt + completion
    started = time.monotonic()
    evaluation = evaluator.evaluate(problem, solution)
    return completion, solution, evaluation, time.monotonic() - started


class BenchmarkRunner:
    def __init__(
        self,
        config: RunConfig,
        provider: Provider,
        evaluator: Evaluator,
        store: EventStore,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.evaluator = evaluator
        self.store = store
        # Frozen for the run's duration: a task's context must not depend on
        # which other tasks a thread pool happened to finish first.
        self.memory_store = memory_store
        self.budget = BudgetManager(config, committed_usd=store.committed_cost())
        self._records_lock = threading.Lock()
        self._records = store.completed_attempts()
        self._stop_reason: str | None = None

    def run(self, problems: Sequence[Problem]) -> dict:
        by_id = {problem.task_id: problem for problem in problems}
        work = [
            (strategy, by_id[task_id])
            for strategy in self.config.strategies
            for task_id in self.config.task_ids
        ]
        self.store.append("run_started", resumed=bool(self._records))
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as executor:
            futures = [
                executor.submit(self._run_trajectory, strategy, problem)
                for strategy, problem in work
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    self._stop_reason = "infrastructure_error"
                    self.store.append(
                        "trajectory_crashed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
        summary = summarize(self.config, self._snapshot_records())
        recorded_cost = float(summary["estimated_cost_usd"])
        summary["unattributed_cost_usd"] = round(
            max(0.0, self.budget.committed_usd - recorded_cost), 8
        )
        summary["estimated_cost_usd"] = round(self.budget.committed_usd, 8)
        summary["stop_reason"] = self._stop_reason
        summary["budget_usd"] = self.config.budget_usd
        summary["budget_committed_usd"] = round(self.budget.committed_usd, 8)
        self.store.write_samples(self._snapshot_records())
        self.store.write_summary(summary)
        self.store.append("run_finished", complete=summary["complete"])
        return summary

    def _run_trajectory(self, strategy: StrategyName, problem: Problem) -> None:
        history = self._history(strategy, problem.task_id)
        if any(item.evaluation.passed for item in history):
            return
        next_attempt = max((item.attempt for item in history), default=0) + 1
        for attempt in range(next_attempt, self.config.max_attempts + 1):
            patch_strategy = strategy in PATCHSEARCH_STRATEGIES
            store = self.memory_store if strategy == "patchsearch-crosstask" else None
            memories, transferred = (
                retrieve_with_store(problem, history, store)
                if patch_strategy
                else ([], [])
            )
            prompt = build_prompt(
                strategy,
                problem,
                attempt,
                history,
                max_attempts=self.config.max_attempts,
                memories=memories,
                transferred=transferred,
            )
            instructions = system_instructions(problem.benchmark)
            reservation = self.budget.acquire(prompt, instructions)
            if reservation is None:
                self._stop_reason = "budget_exhausted"
                self.store.append(
                    "budget_exhausted",
                    task_id=problem.task_id,
                    strategy=strategy,
                    attempt=attempt,
                )
                return
            self.store.append(
                "request_started",
                task_id=problem.task_id,
                strategy=strategy,
                attempt=attempt,
                prompt=prompt,
                reserved_cost_usd=reservation.amount_usd,
            )
            total_started = time.monotonic()
            try:
                response = self.provider.generate(prompt, instructions=instructions)
            except Exception as exc:
                self.budget.release(reservation)
                self._stop_reason = "provider_error"
                self.store.append(
                    "request_failed",
                    task_id=problem.task_id,
                    strategy=strategy,
                    attempt=attempt,
                    error=f"{type(exc).__name__}: {exc}",
                )
                return
            cost = self.budget.reconcile(reservation, response.usage)
            try:
                completion, solution, evaluation, eval_latency = score_output(
                    problem, response.output_text, self.evaluator
                )
            except Exception as exc:
                self._stop_reason = "evaluator_error"
                self.store.append(
                    "evaluation_failed",
                    task_id=problem.task_id,
                    strategy=strategy,
                    attempt=attempt,
                    response_id=response.response_id,
                    request_id=response.request_id,
                    usage=asdict(response.usage),
                    estimated_cost_usd=cost,
                    error=f"{type(exc).__name__}: {exc}",
                )
                return
            record = AttemptRecord(
                run_id=self.config.run_id,
                task_id=problem.task_id,
                strategy=strategy,
                attempt=attempt,
                prompt=prompt,
                completion=completion,
                solution=solution,
                response_id=response.response_id,
                request_id=response.request_id,
                usage=response.usage,
                evaluation=evaluation,
                model_latency_seconds=response.latency_seconds,
                evaluation_latency_seconds=eval_latency,
                total_latency_seconds=time.monotonic() - total_started,
                estimated_cost_usd=cost,
                created_at=utc_now(),
                resolved_model=response.resolved_model,
                patches=(),
                memories=(),
                retrieved_memories=tuple(memories) + tuple(transferred),
            )
            if patch_strategy and not evaluation.passed:
                memory = extract_memory(
                    problem,
                    evaluation.feedback,
                    attempt,
                    include_explanation=strategy == "patchsearch-explained",
                )
                if memory is not None:
                    record = replace(record, memories=(memory,))
            with self._records_lock:
                self._records.append(record)
            self.store.append("attempt_completed", record=record.to_dict())
            history.append(record)
            if evaluation.passed:
                return

    def _history(self, strategy: StrategyName, task_id: str) -> list[AttemptRecord]:
        with self._records_lock:
            return sorted(
                [
                    item
                    for item in self._records
                    if item.strategy == strategy and item.task_id == task_id
                ],
                key=lambda item: item.attempt,
            )

    def _snapshot_records(self) -> list[AttemptRecord]:
        with self._records_lock:
            return list(self._records)
