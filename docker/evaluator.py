from __future__ import annotations

import contextlib
import io
import json
import multiprocessing
from queue import Empty
import sys
from typing import Any

from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import check_correctness, get_groundtruth
from evalplus.eval.utils import reliability_guard, swallow_io

MAX_REPR = 2048


def clipped_repr(value: Any) -> str:
    try:
        rendered = repr(value)
    except BaseException:
        rendered = f"<{type(value).__name__}: repr failed>"
    rendered = rendered.replace("\x00", "\\x00")
    if len(rendered) > MAX_REPR:
        return rendered[: MAX_REPR - 15] + "...<truncated>"
    return rendered


def diagnostic_worker(solution: str, entry_point: str, inp: Any, queue: Any) -> None:
    namespace: dict[str, Any] = {}
    try:
        reliability_guard(maximum_memory_bytes=384 * 1024 * 1024)
        with swallow_io():
            exec(compile(solution, "candidate.py", "exec"), namespace)
        function = namespace.get(entry_point)
        if not callable(function):
            queue.put(
                {
                    "kind": "missing_entry_point",
                    "message": f"candidate did not define callable {entry_point}",
                }
            )
            return
        with swallow_io():
            actual = function(*inp)
        queue.put(
            {
                "kind": "wrong_answer",
                "message": "candidate output did not match the reference oracle",
                "actual_repr": clipped_repr(actual),
            }
        )
    except SyntaxError as exc:
        queue.put(
            {
                "kind": "syntax_error",
                "message": f"{exc.msg} at line {exc.lineno}",
                "exception_type": "SyntaxError",
            }
        )
    except BaseException as exc:
        queue.put(
            {
                "kind": "runtime_error",
                "message": clipped_repr(str(exc)),
                "exception_type": type(exc).__name__,
            }
        )


def diagnose(
    solution: str,
    problem: dict[str, Any],
    inputs: list[Any],
    expected: list[Any],
    fail_index: int,
) -> dict[str, Any]:
    try:
        compile(solution, "candidate.py", "exec")
    except SyntaxError as exc:
        return {
            "kind": "syntax_error",
            "message": f"{exc.msg} at line {exc.lineno}",
            "exception_type": "SyntaxError",
        }
    if not inputs:
        return {
            "kind": "runtime_error",
            "message": "candidate failed before any test completed",
        }
    index = min(max(fail_index, 0), len(inputs) - 1)
    queue: Any = multiprocessing.Queue(maxsize=1)
    process = multiprocessing.Process(
        target=diagnostic_worker,
        args=(solution, problem["entry_point"], inputs[index], queue),
    )
    process.start()
    process.join(timeout=3.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)
        diagnostic = {
            "kind": "timeout",
            "message": "candidate timed out on the first failing input",
        }
    else:
        try:
            diagnostic = queue.get(timeout=0.2)
        except Empty:
            diagnostic = {
                "kind": "runtime_error",
                "message": "candidate process exited without a diagnostic",
            }
    diagnostic["input_repr"] = clipped_repr(inputs[index])
    if diagnostic["kind"] == "wrong_answer":
        diagnostic["expected_repr"] = clipped_repr(expected[index])
    return diagnostic


def evaluate(job: dict[str, Any]) -> dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        problems = get_human_eval_plus(version="v0.1.10")
    task_id = job.get("task_id")
    if task_id not in problems:
        raise ValueError(f"unknown task id: {task_id}")
    solution = job.get("solution")
    if not isinstance(solution, str) or not solution:
        raise ValueError("solution must be a non-empty string")
    problem = problems[task_id]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        groundtruth = get_groundtruth(
            problems, get_human_eval_plus_hash(version="v0.1.10"), []
        )[task_id]
    checked = check_correctness(
        "humaneval",
        0,
        problem,
        solution,
        groundtruth,
        base_only=False,
        fast_check=True,
    )
    base_status, base_details = checked["base"]
    plus_status, plus_details = checked["plus"]
    passed = base_status == "pass" and plus_status == "pass"
    feedback = None
    if not passed:
        if base_status != "pass":
            inputs = problem["base_input"]
            expected = groundtruth["base"]
            details = base_details
        else:
            inputs = problem["plus_input"]
            expected = groundtruth["plus"]
            details = plus_details
        fail_index = len(details) - 1 if details else 0
        feedback = diagnose(solution, problem, inputs, expected, fail_index)
    return {
        "base_status": base_status,
        "plus_status": plus_status,
        "passed": passed,
        "feedback": feedback,
    }


def main() -> int:
    try:
        job = json.load(sys.stdin)
        result = evaluate(job)
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
