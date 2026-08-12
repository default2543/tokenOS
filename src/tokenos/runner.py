from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
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
from tokenos.strategies import build_prompt, parse_completion


class Provider(Protocol):
    def generate(self, prompt: str) -> ModelResponse: ...


class Evaluator(Protocol):
    def evaluate(self, problem: Problem, solution: str) -> EvaluationResult: ...


class BenchmarkRunner:
    def __init__(
        self,
        config: RunConfig,
        provider: Provider,
        evaluator: Evaluator,
        store: EventStore,
    ) -> None:
        self.config = config
        self.provider = provider
        self.evaluator = evaluator
        self.store = store
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
            prompt = build_prompt(
                strategy,
                problem,
                attempt,
                history,
                max_attempts=self.config.max_attempts,
            )
            reservation = self.budget.acquire(prompt)
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
                response = self.provider.generate(prompt)
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
                completion = parse_completion(response.output_text)
                solution = problem.prompt + completion
                eval_started = time.monotonic()
                evaluation = self.evaluator.evaluate(problem, solution)
                eval_latency = time.monotonic() - eval_started
            except ValueError as exc:
                completion = response.output_text
                solution = problem.prompt
                eval_latency = 0.0
                evaluation = EvaluationResult(
                    base_status="fail",
                    plus_status="fail",
                    passed=False,
                    feedback=FailureFeedback(
                        kind="invalid_output", message=str(exc)
                    ),
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
            )
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
