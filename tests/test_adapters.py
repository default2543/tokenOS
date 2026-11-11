import pytest

from tokenos.adapters import (
    AdapterError,
    PythonBenchmarkAdapter,
    get_benchmark_adapter,
    register_benchmark_adapter,
)
from tokenos.models import Problem


def test_humaneval_adapter_normalizes_typed_oracle_evidence() -> None:
    problem = Problem("HumanEval/0", "def f(x):\n", "f")
    result = get_benchmark_adapter("humaneval-plus").decode_result(
        problem,
        {
            "base_status": "fail",
            "plus_status": "fail",
            "passed": False,
            "evidence": {
                "kind": "wrong_answer",
                "message": "mismatch",
                "provenance": "oracle",
                "observations": [
                    {"kind": "function_call", "value": "(1,)", "trusted": True},
                    {"kind": "expected_value", "value": "2", "trusted": True},
                    {"kind": "actual_value", "value": "1"},
                ],
            },
        },
    )
    assert result.feedback is not None
    assert result.feedback.benchmark == "humaneval-plus"
    assert [item.kind for item in result.feedback.observations] == [
        "function_call", "expected_value", "actual_value"
    ]


def test_bigcodebench_adapter_bounds_trace_and_preserves_test_identity() -> None:
    problem = Problem("BigCodeBench/1", "def f():\n", "f", benchmark="bigcodebench-hard")
    result = get_benchmark_adapter("bigcodebench-hard").decode_result(
        problem,
        {
            "base_status": "fail", "plus_status": "fail", "passed": False,
            "evidence": {
                "kind": "runtime_error", "message": "boom\x00 now",
                "test_id": "test_case", "observations": [
                    {"kind": "traceback", "value": "x" * 4000},
                    {"kind": "exception", "value": "ValueError"},
                ],
            },
        },
    )
    evidence = result.feedback
    assert evidence is not None and evidence.truncated
    assert evidence.test_id == "test_case"
    assert "\x00" not in evidence.message
    assert max(len(item.value) for item in evidence.observations) <= 2000


@pytest.mark.parametrize("payload", [[], {}, {"passed": "no", "base_status": "fail", "plus_status": "fail"}])
def test_adapter_rejects_malformed_evaluator_results(payload: object) -> None:
    problem = Problem("X/0", "def f():\n", "f")
    with pytest.raises(AdapterError):
        get_benchmark_adapter("humaneval-plus").decode_result(problem, payload)


def test_public_adapter_registration_supports_new_benchmarks() -> None:
    adapter = PythonBenchmarkAdapter("custom-python")
    register_benchmark_adapter(adapter)
    assert get_benchmark_adapter("custom-python") is adapter
    job = adapter.build_job(
        Problem("Custom/0", "def f():\n", "f", benchmark="custom-python"), "def f(): pass"
    )
    assert job["schema_version"] == 2
