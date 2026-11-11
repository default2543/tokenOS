"""Evaluator viability preflight.

A benchmark task is only evidence about a memory strategy if its own reference
solution passes inside the evaluator image. BigCodeBench-Hard tasks routinely
fail for environmental reasons instead: they shell out to binaries the image
does not ship, or reach a network the sandbox denies. Those tasks contribute
noise to every strategy, so they are measured once here and excluded from
comparisons rather than being paid for on every run.

Preflight spends no model tokens.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Protocol, Sequence

from tokenos.models import EvaluationResult, Problem

PREFLIGHT_ROOT = Path(".tokenos/preflight")
PREFLIGHT_SCHEMA_VERSION = 1
MAX_DETAIL_CHARS = 300


class PreflightError(RuntimeError):
    pass


class Evaluator(Protocol):
    def evaluate(self, problem: Problem, solution: str) -> EvaluationResult: ...


@dataclass(frozen=True, slots=True)
class TaskViability:
    task_id: str
    viable: bool
    status: str
    detail: str = ""


def preflight_path(
    dataset_name: str, dataset_version: str, root: Path = PREFLIGHT_ROOT
) -> Path:
    return root / f"{dataset_name}-{dataset_version}.json"


def check_task(
    problem: Problem, canonical_solution: str, evaluator: Evaluator
) -> TaskViability:
    """Run one reference solution through the evaluator image."""
    if not canonical_solution.strip():
        return TaskViability(
            problem.task_id, False, "missing_reference", "dataset has no reference body"
        )
    try:
        result = evaluator.evaluate(problem, problem.prompt + canonical_solution)
    except Exception as exc:
        return TaskViability(
            problem.task_id,
            False,
            "evaluator_error",
            f"{type(exc).__name__}: {exc}"[:MAX_DETAIL_CHARS],
        )
    if result.passed:
        return TaskViability(problem.task_id, True, "pass")
    detail = ""
    if result.feedback is not None:
        detail = f"{result.feedback.kind}: {result.feedback.message}"[:MAX_DETAIL_CHARS]
    return TaskViability(problem.task_id, False, "reference_failed", detail)


def run_preflight(
    problems: Sequence[Problem],
    canonical_solutions: dict[str, str],
    evaluator: Evaluator,
    *,
    dataset_name: str,
    dataset_version: str,
    dataset_sha256: str,
    evaluator_image: str,
    concurrency: int = 4,
) -> dict:
    """Check every problem's reference solution and build a viability report."""
    missing = [p.task_id for p in problems if p.task_id not in canonical_solutions]
    if missing:
        raise PreflightError(
            f"dataset has no reference solution for: {', '.join(missing[:5])}"
        )
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        results = list(
            executor.map(
                lambda problem: check_task(
                    problem, canonical_solutions[problem.task_id], evaluator
                ),
                problems,
            )
        )
    results.sort(key=lambda item: item.task_id)
    viable = [item.task_id for item in results if item.viable]
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "dataset_sha256": dataset_sha256,
        "evaluator_image": evaluator_image,
        "checked_tasks": len(results),
        "viable_tasks": len(viable),
        "viable_task_ids": viable,
        "results": [asdict(item) for item in results],
    }


def write_report(report: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_report(path: Path) -> dict:
    if not path.is_file():
        raise PreflightError(
            f"preflight report not found: {path}; run 'tokenos preflight' first"
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"invalid preflight report {path}: {exc}") from exc
    if report.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise PreflightError(
            f"preflight report {path} uses an unsupported schema version"
        )
    return report


def viable_task_ids(
    path: Path, *, dataset_sha256: str, evaluator_image: str
) -> set[str]:
    """Read a report, refusing one measured against a different environment."""
    report = read_report(path)
    if report.get("dataset_sha256") != dataset_sha256:
        raise PreflightError(
            f"preflight report {path} was measured against a different dataset; "
            "re-run 'tokenos preflight'"
        )
    if report.get("evaluator_image") != evaluator_image:
        raise PreflightError(
            f"preflight report {path} was measured against evaluator image "
            f"{report.get('evaluator_image')!r}, not {evaluator_image!r}; "
            "re-run 'tokenos preflight'"
        )
    return set(report.get("viable_task_ids", ()))
