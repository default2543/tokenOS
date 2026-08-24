from __future__ import annotations

import json
import contextlib
import io
import multiprocessing
import os
import re
import sys
import tempfile
import traceback
from typing import Any
import unittest

MAX_MESSAGE = 2000
TASK_TIMEOUT_SECONDS = 30


def sanitize_message(value: object) -> str:
    message = " ".join(str(value).replace("\x00", "\\x00").split())
    if len(message) > MAX_MESSAGE:
        return message[: MAX_MESSAGE - 15].rstrip() + "...<truncated>"
    return message


def failure_feedback(status: str, details: dict[str, str]) -> dict[str, Any]:
    combined = "\n".join(details.values())
    message = sanitize_message(combined or f"BigCodeBench evaluator returned {status}")
    if status == "timeout":
        return {
            "kind": "timeout",
            "message": "candidate exceeded the BigCodeBench task time limit",
            "observations": [{"kind": "timeout", "value": "30 seconds"}],
            "provenance": "bigcodebench-unittest",
        }
    test_id = next(iter(details), None)
    observations: list[dict[str, Any]] = []
    if test_id:
        observations.append({"kind": "test_name", "value": test_id})
    if combined:
        observations.append({"kind": "traceback", "value": sanitize_message(combined)})
    syntax = re.search(r"(SyntaxError|IndentationError):\s*([^\n]+)", combined)
    if syntax:
        return {
            "kind": "syntax_error",
            "message": sanitize_message(syntax.group(2)),
            "observations": observations + [
                {"kind": "exception", "value": syntax.group(1)}
            ],
            "test_id": test_id,
            "provenance": "bigcodebench-unittest",
        }
    assertion = re.search(r"AssertionError(?::\s*([^\n]+))?", combined)
    if assertion:
        actual = sanitize_message(assertion.group(1)) if assertion.group(1) else None
        feedback: dict[str, Any] = {
            "kind": "wrong_answer",
            "message": message,
            "observations": observations,
            "test_id": test_id,
            "provenance": "bigcodebench-unittest",
        }
        if actual:
            feedback["observations"].append({"kind": "actual_value", "value": actual})
        return feedback
    exception = re.findall(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):\s*(.*)$", combined, re.M)
    if exception:
        exception_type, exception_message = exception[-1]
        return {
            "kind": "runtime_error",
            "message": sanitize_message(exception_message),
            "observations": observations + [{
                "kind": "exception",
                "value": exception_type.rsplit(".", 1)[-1],
            }],
            "test_id": test_id,
            "provenance": "bigcodebench-unittest",
        }
    return {
        "kind": "runtime_error",
        "message": message,
        "observations": observations,
        "test_id": test_id,
        "provenance": "bigcodebench-unittest",
    }


def test_worker(
    solution: str, test_code: str, connection: multiprocessing.connection.Connection
) -> None:
    namespace: dict[str, Any] = {"__name__": "__test__"}
    try:
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                exec(compile(solution + "\n" + test_code, "__test__.py", "exec"), namespace)
                test_cases = namespace.get("TestCases")
                if not isinstance(test_cases, type):
                    raise RuntimeError("test code did not define TestCases")
                suite = unittest.TestLoader().loadTestsFromTestCase(test_cases)
                result = unittest.TestResult()
                suite.run(result)
        issues = result.failures + result.errors
        details = {
            test.id().rsplit(".", 1)[-1]: trace for test, trace in issues
        }
        connection.send(("pass" if not details else "fail", details))
    except BaseException:
        connection.send(("fail", {"ALL": traceback.format_exc()}))
    finally:
        connection.close()


def run_tests(solution: str, test_code: str) -> tuple[str, dict[str, str]]:
    receiver, sender = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(
        target=test_worker, args=(solution, test_code, sender)
    )
    process.start()
    sender.close()
    try:
        if receiver.poll(TASK_TIMEOUT_SECONDS):
            return receiver.recv()
        return "timeout", {}
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=1)
        process.close()
        receiver.close()


def evaluate(job: dict[str, Any]) -> dict[str, Any]:
    solution = job.get("solution")
    entry_point = job.get("entry_point")
    test_code = job.get("test_code")
    if not isinstance(solution, str) or not solution:
        raise ValueError("solution must be a non-empty string")
    if not isinstance(entry_point, str) or not entry_point.isidentifier():
        raise ValueError("entry_point must be an identifier")
    if not isinstance(test_code, str) or not test_code:
        raise ValueError("test_code must be a non-empty string")
    try:
        compile(solution, "candidate.py", "exec")
    except SyntaxError as exc:
        feedback = {
            "kind": "syntax_error",
            "message": sanitize_message(f"{exc.msg} at line {exc.lineno}"),
            "observations": [{"kind": "exception", "value": "SyntaxError"}],
            "provenance": "bigcodebench-unittest",
        }
        return {
            "base_status": "fail",
            "plus_status": "fail",
            "passed": False,
            "evidence": feedback,
        }
    status, details = run_tests(solution, test_code)
    passed = status == "pass"
    return {
        "base_status": status,
        "plus_status": status,
        "passed": passed,
        "evidence": None if passed else failure_feedback(status, details),
    }


def main() -> int:
    try:
        result = evaluate(json.load(sys.stdin))
    except BaseException as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
