"""The TokenOS runtime: one task, one budget, the measured strategy.

The benchmark runner compares strategies across a dataset. The runtime is what
the comparison was for: it solves a single task with the strategy that won,
under a context budget the caller chooses.

    runtime = Runtime(context_budget=8000)
    result = runtime.run(problem)

Each attempt flows through the same pieces the benchmark measured -- task-local
retrieval, optional cross-task allocation from a frozen store, then the context
compiler, which enforces the budget -- and each failure is distilled into
bounded memory for the next attempt. With no provider or evaluator supplied,
the runtime uses Azure OpenAI and the Docker sandbox, configured from the same
environment variables as `tokenos benchmark`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
import time
from typing import Mapping

from tokenos.budget import BudgetManager
from tokenos.compiler import CompiledContext, compile_context
from tokenos.config import (
    ConfigurationError,
    config_from_environment,
    new_run_id,
    utc_now,
)
from tokenos.evaluator import DockerEvaluator
from tokenos.models import AttemptRecord, Problem, RunConfig, StrategyName
from tokenos.patchsearch.extractor import extract_memory
from tokenos.patchsearch.retriever import retrieve_with_store
from tokenos.provider import AzureOpenAIProvider
from tokenos.runner import Evaluator, Provider, score_output
from tokenos.store import MemoryStore
from tokenos.strategies import PATCHSEARCH_STRATEGIES, system_instructions

DEFAULT_STRATEGY: StrategyName = "patchsearch-adaptive"
DEFAULT_CONTEXT_BUDGET = 8000


@dataclass(frozen=True, slots=True)
class RuntimeAttempt:
    """One attempt and the context that was compiled for it."""

    record: AttemptRecord
    context: CompiledContext

    @property
    def passed(self) -> bool:
        return self.record.evaluation.passed


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    task_id: str
    strategy: StrategyName
    context_budget: int | None
    attempts: tuple[RuntimeAttempt, ...] = ()
    stop_reason: str | None = None

    @property
    def passed(self) -> bool:
        return any(attempt.passed for attempt in self.attempts)

    @property
    def solution(self) -> str | None:
        """The passing solution, or the last attempt's when none passed."""
        if not self.attempts:
            return None
        for attempt in self.attempts:
            if attempt.passed:
                return attempt.record.solution
        return self.attempts[-1].record.solution

    @property
    def input_tokens(self) -> int:
        return sum(item.record.usage.input_tokens for item in self.attempts)

    @property
    def output_tokens(self) -> int:
        return sum(item.record.usage.output_tokens for item in self.attempts)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        return sum(item.record.estimated_cost_usd for item in self.attempts)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "strategy": self.strategy,
            "context_budget": self.context_budget,
            "passed": self.passed,
            "stop_reason": self.stop_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 8),
            "attempts": [
                {
                    "attempt": item.record.attempt,
                    "passed": item.passed,
                    "failure_kind": (
                        item.record.evaluation.feedback.kind
                        if item.record.evaluation.feedback is not None
                        else None
                    ),
                    "input_tokens": item.record.usage.input_tokens,
                    "output_tokens": item.record.usage.output_tokens,
                    "context": item.context.to_dict(),
                }
                for item in self.attempts
            ],
        }


def _local_config(strategy: StrategyName, max_attempts: int) -> RunConfig:
    """A config for injected providers, where there are no prices to guard."""
    return RunConfig(
        run_id=new_run_id(),
        strategies=(strategy,),
        task_ids=(),
        dataset_path="",
        max_attempts=max_attempts,
        concurrency=1,
        budget_usd=math.inf,
        created_at=utc_now(),
    )


class Runtime:
    """Solve one task at a time under a fixed context budget."""

    def __init__(
        self,
        context_budget: int | None = DEFAULT_CONTEXT_BUDGET,
        *,
        provider: Provider | None = None,
        evaluator: Evaluator | None = None,
        strategy: StrategyName = DEFAULT_STRATEGY,
        max_attempts: int = 5,
        memory_store: MemoryStore | None = None,
        budget_usd: float = 1.0,
        evaluator_image: str | None = None,
        config: RunConfig | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if strategy not in PATCHSEARCH_STRATEGIES:
            raise ConfigurationError(
                f"the runtime uses bounded memory; {strategy} is a benchmark "
                "baseline, not a runtime strategy"
            )
        if context_budget is not None and context_budget <= 0:
            raise ConfigurationError("context_budget must be positive")
        if max_attempts < 1:
            raise ConfigurationError("max_attempts must be positive")
        if config is None:
            if provider is None:
                # Real calls cost money, so they carry a real spend guard.
                config = config_from_environment(
                    strategies=(strategy,),
                    task_ids=(),
                    dataset_path=os.devnull,
                    dataset_sha256="",
                    max_attempts=max_attempts,
                    concurrency=1,
                    budget_usd=budget_usd,
                    env=env,
                )
            else:
                config = _local_config(strategy, max_attempts)
        config = replace(config, max_attempts=max_attempts)
        if evaluator_image is not None:
            config = replace(config, evaluator_image=evaluator_image)
        if provider is None:
            provider = AzureOpenAIProvider(config)
        if evaluator is None:
            evaluator = DockerEvaluator(config)
        self.config = config
        self.context_budget = context_budget
        self.strategy = strategy
        self.provider = provider
        self.evaluator = evaluator
        self.memory_store = memory_store
        self.budget = BudgetManager(config)

    def compile(
        self, problem: Problem, history: list[AttemptRecord], attempt: int
    ) -> CompiledContext:
        """Build the context the next attempt would see, without calling a model."""
        memories, transferred = retrieve_with_store(
            problem, history, self.memory_store
        )
        return compile_context(
            self.strategy,
            problem,
            attempt,
            history,
            memories=memories,
            transferred=transferred,
            budget_tokens=self.context_budget,
            instructions=system_instructions(problem.benchmark),
            max_attempts=self.config.max_attempts,
        )

    def run(self, problem: Problem) -> RuntimeResult:
        history: list[AttemptRecord] = []
        attempts: list[RuntimeAttempt] = []
        instructions = system_instructions(problem.benchmark)
        stop_reason: str | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            context = self.compile(problem, history, attempt)
            reservation = self.budget.acquire(context.prompt, instructions)
            if reservation is None:
                stop_reason = "budget_exhausted"
                break
            started = time.monotonic()
            try:
                response = self.provider.generate(
                    context.prompt, instructions=instructions
                )
            except Exception:
                self.budget.release(reservation)
                raise
            cost = self.budget.reconcile(reservation, response.usage)
            completion, solution, evaluation, eval_latency = score_output(
                problem, response.output_text, self.evaluator
            )
            record = AttemptRecord(
                run_id=self.config.run_id,
                task_id=problem.task_id,
                strategy=self.strategy,
                attempt=attempt,
                prompt=context.prompt,
                completion=completion,
                solution=solution,
                response_id=response.response_id,
                request_id=response.request_id,
                usage=response.usage,
                evaluation=evaluation,
                model_latency_seconds=response.latency_seconds,
                evaluation_latency_seconds=eval_latency,
                total_latency_seconds=time.monotonic() - started,
                estimated_cost_usd=cost,
                created_at=utc_now(),
                resolved_model=response.resolved_model,
                retrieved_memories=context.memories + context.transferred,
            )
            if not evaluation.passed:
                memory = extract_memory(
                    problem,
                    evaluation.feedback,
                    attempt,
                    include_explanation=self.strategy == "patchsearch-explained",
                )
                if memory is not None:
                    record = replace(record, memories=(memory,))
            history.append(record)
            attempts.append(RuntimeAttempt(record=record, context=context))
            if evaluation.passed:
                break
        else:
            stop_reason = "max_attempts"
        return RuntimeResult(
            task_id=problem.task_id,
            strategy=self.strategy,
            context_budget=self.context_budget,
            attempts=tuple(attempts),
            stop_reason=stop_reason,
        )
