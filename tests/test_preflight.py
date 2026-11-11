import pytest

from tokenos.models import EvaluationResult, FailureFeedback, Problem
from tokenos.preflight import (
    PreflightError,
    check_task,
    preflight_path,
    read_report,
    run_preflight,
    viable_task_ids,
    write_report,
)

VIABLE = Problem("B/1", "def task_func():\n", "task_func", "bigcodebench-hard")
BROKEN = Problem("B/2", "def task_func():\n", "task_func", "bigcodebench-hard")
SOLUTIONS = {"B/1": "    return 1\n", "B/2": "    return 2\n"}


class FakeEvaluator:
    """Passes only the task whose reference solution the image can run."""

    def __init__(self, failing: set[str] = frozenset(), raising: set[str] = frozenset()):
        self.failing = failing
        self.raising = raising
        self.seen: list[tuple[str, str]] = []

    def evaluate(self, problem: Problem, solution: str) -> EvaluationResult:
        self.seen.append((problem.task_id, solution))
        if problem.task_id in self.raising:
            raise RuntimeError("evaluator container exited with 1")
        if problem.task_id in self.failing:
            return EvaluationResult(
                "fail",
                "fail",
                False,
                FailureFeedback(
                    kind="runtime_error",
                    message="[Errno 2] No such file or directory: 'wget'",
                ),
            )
        return EvaluationResult("pass", "pass", True)


def _report(evaluator, problems=(VIABLE, BROKEN), **overrides) -> dict:
    arguments = {
        "dataset_name": "bigcodebench-hard",
        "dataset_version": "v0.1.4",
        "dataset_sha256": "abc",
        "evaluator_image": "img:1",
        "concurrency": 2,
    }
    arguments.update(overrides)
    return run_preflight(list(problems), SOLUTIONS, evaluator, **arguments)


def test_reference_solution_is_appended_to_the_prompt() -> None:
    evaluator = FakeEvaluator()
    check_task(VIABLE, SOLUTIONS["B/1"], evaluator)
    assert evaluator.seen == [("B/1", "def task_func():\n    return 1\n")]


def test_task_whose_reference_solution_fails_is_not_viable() -> None:
    result = check_task(BROKEN, SOLUTIONS["B/2"], FakeEvaluator(failing={"B/2"}))
    assert not result.viable
    assert result.status == "reference_failed"
    assert "wget" in result.detail


def test_evaluator_crash_is_recorded_rather_than_raised() -> None:
    result = check_task(BROKEN, SOLUTIONS["B/2"], FakeEvaluator(raising={"B/2"}))
    assert not result.viable
    assert result.status == "evaluator_error"


def test_missing_reference_body_is_not_viable() -> None:
    result = check_task(VIABLE, "   \n", FakeEvaluator())
    assert not result.viable
    assert result.status == "missing_reference"


def test_report_separates_viable_from_environment_broken_tasks() -> None:
    report = _report(FakeEvaluator(failing={"B/2"}))
    assert report["checked_tasks"] == 2
    assert report["viable_tasks"] == 1
    assert report["viable_task_ids"] == ["B/1"]


def test_report_requires_a_reference_solution_for_every_task() -> None:
    unknown = Problem("B/9", "def task_func():\n", "task_func", "bigcodebench-hard")
    with pytest.raises(PreflightError, match="no reference solution"):
        _report(FakeEvaluator(), problems=(unknown,))


def test_report_round_trips_through_disk(tmp_path) -> None:
    path = write_report(_report(FakeEvaluator(failing={"B/2"})), tmp_path / "r.json")
    assert read_report(path)["viable_task_ids"] == ["B/1"]
    assert viable_task_ids(path, dataset_sha256="abc", evaluator_image="img:1") == {"B/1"}


def test_report_measured_against_another_dataset_is_refused(tmp_path) -> None:
    path = write_report(_report(FakeEvaluator()), tmp_path / "r.json")
    with pytest.raises(PreflightError, match="different dataset"):
        viable_task_ids(path, dataset_sha256="other", evaluator_image="img:1")


def test_report_measured_against_another_image_is_refused(tmp_path) -> None:
    # Viability is a property of the image, so a report from a different one
    # cannot be reused.
    path = write_report(_report(FakeEvaluator()), tmp_path / "r.json")
    with pytest.raises(PreflightError, match="evaluator image"):
        viable_task_ids(path, dataset_sha256="abc", evaluator_image="other:2")


def test_missing_report_names_the_command_that_creates_it(tmp_path) -> None:
    with pytest.raises(PreflightError, match="tokenos preflight"):
        read_report(tmp_path / "absent.json")


def test_report_path_is_scoped_to_dataset_and_version() -> None:
    path = preflight_path("bigcodebench-hard", "v0.1.4")
    assert path.name == "bigcodebench-hard-v0.1.4.json"
