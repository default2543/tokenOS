"""Unit tests for the BigCodeBench evaluator's evidence parsing.

The evaluator script ships inside the container image rather than the package,
so it is loaded by path. These tests need no Docker daemon.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "docker" / "bigcodebench_evaluator.py"


def _load():
    spec = importlib.util.spec_from_file_location("bigcodebench_evaluator", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluator = _load()


def _observations(feedback) -> dict[str, str]:
    return {item["kind"]: item["value"] for item in feedback["observations"]}


FTP_TRACE = """Traceback (most recent call last):
  File "__test__.py", line 132, in test_case_2
AssertionError: 'Failed to connect' != 'Failed to connect to invalid_server'
- Failed to connect
+ Failed to connect to invalid_server
"""

OTHER_TRACE = """Traceback (most recent call last):
  File "__test__.py", line 200, in test_case_9
AssertionError: 1 != 2
"""


def test_assertion_message_is_recorded_as_a_diff_not_a_returned_value() -> None:
    feedback = evaluator.failure_feedback("fail", {"test_case_2": FTP_TRACE})
    assert feedback["kind"] == "wrong_answer"
    observations = _observations(feedback)
    assert "actual_value" not in observations
    assert "Failed to connect to invalid_server" in observations["assertion_diff"]


def test_assertion_diff_does_not_bleed_into_another_test_traceback() -> None:
    # `details` holds every failing test; the diff must describe only the test
    # that `test_id` names.
    feedback = evaluator.failure_feedback(
        "fail", {"test_case_2": FTP_TRACE, "test_case_9": OTHER_TRACE}
    )
    assert feedback["test_id"] == "test_case_2"
    diff = _observations(feedback)["assertion_diff"]
    assert "test_case_9" not in diff
    assert "1 != 2" not in diff


def test_full_traceback_is_still_available_for_full_history() -> None:
    feedback = evaluator.failure_feedback(
        "fail", {"test_case_2": FTP_TRACE, "test_case_9": OTHER_TRACE}
    )
    assert "test_case_9" in _observations(feedback)["traceback"]


def test_runtime_error_keeps_its_exception_type() -> None:
    trace = """Traceback (most recent call last):
  File "__test__.py", line 4, in test_case_1
FileNotFoundError: [Errno 2] No such file or directory: 'wget'
"""
    feedback = evaluator.failure_feedback("fail", {"test_case_1": trace})
    assert feedback["kind"] == "runtime_error"
    assert _observations(feedback)["exception"] == "FileNotFoundError"


def test_timeout_reports_no_assertion_evidence() -> None:
    feedback = evaluator.failure_feedback("timeout", {})
    assert feedback["kind"] == "timeout"
    assert "assertion_diff" not in _observations(feedback)


def test_bare_assertion_without_a_message_yields_no_diff() -> None:
    trace = """Traceback (most recent call last):
  File "__test__.py", line 4, in test_case_1
AssertionError
"""
    feedback = evaluator.failure_feedback("fail", {"test_case_1": trace})
    assert feedback["kind"] == "wrong_answer"
    assert "assertion_diff" not in _observations(feedback)


@pytest.mark.parametrize("limit", ["ASSERTION_REPR_LIMIT", "ASSERTION_DIFF_LIMIT"])
def test_repr_bounds_stay_under_the_message_cap(limit: str) -> None:
    # Widened so unittest stops eliding values, but still bounded.
    assert 640 < getattr(evaluator, limit)
    assert getattr(evaluator, limit) <= evaluator.MAX_MESSAGE * 2


def test_worker_that_dies_without_reporting_is_a_failure_not_an_eof_crash() -> None:
    class DeadPipe:
        def poll(self, timeout):
            return True  # at EOF a closed pipe reports readable

        def recv(self):
            raise EOFError

        def close(self):
            pass

    class DeadProcess:
        def start(self): ...
        def is_alive(self):
            return False
        def join(self, timeout=None): ...
        def close(self): ...

    class Sender:
        def close(self): ...

    original_pipe = evaluator.multiprocessing.Pipe
    original_process = evaluator.multiprocessing.Process
    evaluator.multiprocessing.Pipe = lambda duplex=False: (DeadPipe(), Sender())
    evaluator.multiprocessing.Process = lambda target, args: DeadProcess()
    try:
        status, details = evaluator.run_tests("x = 1", "y = 2")
    finally:
        evaluator.multiprocessing.Pipe = original_pipe
        evaluator.multiprocessing.Process = original_process
    assert status == "fail"
    assert "without reporting" in details["ALL"]


def test_subprocess_output_does_not_corrupt_the_result_json(tmp_path) -> None:
    # A candidate that shells out writes to file descriptor 1, which carries
    # this evaluator's result JSON.
    import json as _json
    import subprocess
    import sys

    solution = (
        "import subprocess, sys\n"
        "def task_func():\n"
        "    subprocess.run([sys.executable, '-c', \"print('LEAKED STDOUT')\"])\n"
        "    return 1\n"
    )
    test_code = (
        "import unittest\n"
        "class TestCases(unittest.TestCase):\n"
        "    def test_case_1(self):\n"
        "        self.assertEqual(task_func(), 1)\n"
    )
    job = {
        "schema_version": 2,
        "task_id": "T/1",
        "solution": solution,
        "entry_point": "task_func",
        "test_code": test_code,
        "libraries": [],
    }
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=_json.dumps(job),
        capture_output=True,
        text=True,
    )
    assert "LEAKED STDOUT" not in completed.stdout
    assert _json.loads(completed.stdout)["passed"] is True
