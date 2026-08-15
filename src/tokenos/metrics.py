from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from tokenos.models import AttemptRecord, RunConfig


def summarize(config: RunConfig, records: Iterable[AttemptRecord]) -> dict[str, Any]:
    items = list(records)
    result: dict[str, Any] = {
        "run_id": config.run_id,
        "planned_tasks": len(config.task_ids),
        "strategies": {},
        "estimated_cost_usd": round(sum(r.estimated_cost_usd for r in items), 8),
    }
    complete = True
    solved_by_strategy: dict[str, set[str]] = {}
    attempted_by_strategy: dict[str, set[str]] = {}
    for strategy in config.strategies:
        strategy_records = [r for r in items if r.strategy == strategy]
        task_records: dict[str, list[AttemptRecord]] = {}
        for record in strategy_records:
            task_records.setdefault(record.task_id, []).append(record)
        attempted = set(task_records)
        solved = {
            task_id
            for task_id, attempts in task_records.items()
            if any(record.evaluation.passed for record in attempts)
        }
        terminal = {
            task_id
            for task_id, attempts in task_records.items()
            if any(record.evaluation.passed for record in attempts)
            or max(record.attempt for record in attempts) >= config.max_attempts
        }
        is_complete = len(terminal) == len(config.task_ids)
        complete = complete and is_complete
        solved_by_strategy[strategy] = solved
        attempted_by_strategy[strategy] = attempted
        usage = {
            "input_tokens": sum(r.usage.input_tokens for r in strategy_records),
            "cached_input_tokens": sum(
                r.usage.cached_input_tokens for r in strategy_records
            ),
            "output_tokens": sum(r.usage.output_tokens for r in strategy_records),
            "reasoning_tokens": sum(
                r.usage.reasoning_tokens for r in strategy_records
            ),
        }
        failures = Counter(
            r.evaluation.feedback.kind
            for r in strategy_records
            if r.evaluation.feedback is not None
        )
        solve_at = {}
        for k in (1, 3, 5):
            solved_at_k = sum(
                1
                for task_id in config.task_ids
                if any(
                    r.task_id == task_id and r.attempt <= k and r.evaluation.passed
                    for r in strategy_records
                )
            )
            solve_at[f"solve@{k}"] = (
                solved_at_k / len(config.task_ids) if config.task_ids else 0.0
            )
        total_tokens = usage["input_tokens"] + usage["output_tokens"]
        retries = [r for r in strategy_records if r.attempt > 1]
        patch_bearing = [r for r in retries if "\nassert " in r.prompt]
        result["strategies"][strategy] = {
            "complete": is_complete,
            "attempted_tasks": len(attempted),
            "terminal_tasks": len(terminal),
            "solved_tasks": len(solved),
            **solve_at,
            "model_calls": len(strategy_records),
            "usage": usage,
            "tokens_per_solved_problem": (
                total_tokens / len(solved) if solved else None
            ),
            "estimated_cost_usd": round(
                sum(r.estimated_cost_usd for r in strategy_records), 8
            ),
            "average_model_latency_seconds": _average(
                r.model_latency_seconds for r in strategy_records
            ),
            "average_evaluation_latency_seconds": _average(
                r.evaluation_latency_seconds for r in strategy_records
            ),
            "failure_distribution": dict(sorted(failures.items())),
            "patches_generated": sum(len(r.patches) for r in strategy_records),
            "patches_retrieved": sum(
                prompt.count("\nassert ")
                for prompt in (r.prompt for r in strategy_records)
            ),
            "retry_model_calls": len(retries),
            "retry_input_tokens": sum(r.usage.input_tokens for r in retries),
            "patch_bearing_attempts": len(patch_bearing),
            "patch_bearing_input_tokens": sum(
                r.usage.input_tokens for r in patch_bearing
            ),
        }
    paired = set(config.task_ids)
    for strategy in config.strategies:
        paired &= attempted_by_strategy.get(strategy, set())
    result["complete"] = complete
    result["paired_attempted_tasks"] = len(paired)
    if len(config.strategies) == 2:
        first, second = config.strategies
        result["paired_solve_delta"] = {
            "comparison": f"{second} minus {first}",
            "tasks": len(paired),
            "delta_solved": len(solved_by_strategy[second] & paired)
            - len(solved_by_strategy[first] & paired),
        }
    return result


def _average(values: Iterable[float]) -> float | None:
    items = list(values)
    return round(sum(items) / len(items), 6) if items else None
