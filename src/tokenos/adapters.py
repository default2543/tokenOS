from __future__ import annotations

from typing import Any, Protocol

from tokenos.models import (
    EvidenceObservation,
    EvaluationResult,
    FailureEvidence,
    Problem,
)

MAX_EVIDENCE_CHARS = 2000
MAX_OBSERVATIONS = 12
_OBSERVATION_KINDS = {
    "function_call",
    "expected_value",
    "actual_value",
    "assertion_diff",
    "exception",
    "timeout",
    "stdout_mismatch",
    "stderr",
    "property",
    "test_name",
    "traceback",
}


class AdapterError(ValueError):
    pass


class BenchmarkAdapter(Protocol):
    """Public extension point for benchmark evaluator protocols."""

    name: str

    def build_job(self, problem: Problem, solution: str) -> dict[str, Any]: ...

    def decode_result(self, problem: Problem, payload: object) -> EvaluationResult: ...


def sanitize_evidence_text(value: object, limit: int = MAX_EVIDENCE_CHARS) -> tuple[str, bool]:
    text = " ".join(str(value).replace("\x00", "\\x00").split())
    if len(text) <= limit:
        return text, False
    suffix = "...<truncated>"
    return text[: limit - len(suffix)].rstrip() + suffix, True


class PythonBenchmarkAdapter:
    """Canonical adapter for sandboxed Python coding benchmark evaluators."""

    def __init__(self, name: str) -> None:
        self.name = name

    def build_job(self, problem: Problem, solution: str) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "task_id": problem.task_id,
            "solution": solution,
            "entry_point": problem.entry_point,
            "test_code": problem.test_code,
            "libraries": problem.libraries,
        }

    def decode_result(self, problem: Problem, payload: object) -> EvaluationResult:
        if not isinstance(payload, dict):
            raise AdapterError("evaluator result must be a JSON object")
        required = {"base_status", "plus_status", "passed"}
        if not required <= payload.keys():
            missing = ", ".join(sorted(required - payload.keys()))
            raise AdapterError(f"evaluator result missing fields: {missing}")
        if not isinstance(payload["passed"], bool):
            raise AdapterError("evaluator passed field must be boolean")
        if not isinstance(payload["base_status"], str) or not isinstance(
            payload["plus_status"], str
        ):
            raise AdapterError("evaluator status fields must be strings")
        feedback = payload.get("evidence", payload.get("feedback"))
        evidence = None if feedback is None else self._decode_evidence(problem, feedback)
        if not payload["passed"] and evidence is None:
            raise AdapterError("failed evaluation must include evidence")
        return EvaluationResult(
            base_status=str(payload["base_status"]),
            plus_status=str(payload["plus_status"]),
            passed=payload["passed"],
            feedback=evidence,
        )

    def _decode_evidence(self, problem: Problem, raw: object) -> FailureEvidence:
        if not isinstance(raw, dict):
            raise AdapterError("evaluator evidence must be a JSON object")
        kind = raw.get("kind")
        if kind not in {
            "wrong_answer", "runtime_error", "syntax_error", "missing_entry_point",
            "timeout", "invalid_output",
        }:
            raise AdapterError(f"unsupported evidence kind: {kind!r}")
        if not isinstance(raw.get("message", ""), str):
            raise AdapterError("evidence message must be a string")
        if "truncated" in raw and not isinstance(raw["truncated"], bool):
            raise AdapterError("evidence truncated flag must be boolean")
        message, message_truncated = sanitize_evidence_text(raw.get("message", ""))
        observations: list[EvidenceObservation] = []
        raw_observations = raw.get("observations", ())
        if not isinstance(raw_observations, (list, tuple)):
            raise AdapterError("evidence observations must be an array")
        if len(raw_observations) > MAX_OBSERVATIONS:
            raise AdapterError(f"evidence exceeds {MAX_OBSERVATIONS} observations")
        for item in raw_observations:
            if not isinstance(item, dict) or item.get("kind") not in _OBSERVATION_KINDS:
                raise AdapterError("evidence contains an invalid observation")
            if not isinstance(item.get("value", ""), str):
                raise AdapterError("observation value must be a string")
            if "trusted" in item and not isinstance(item["trusted"], bool):
                raise AdapterError("observation trusted flag must be boolean")
            if "truncated" in item and not isinstance(item["truncated"], bool):
                raise AdapterError("observation truncated flag must be boolean")
            value, clipped = sanitize_evidence_text(item.get("value", ""))
            observations.append(
                EvidenceObservation(
                    kind=item["kind"],
                    value=value,
                    trusted=bool(item.get("trusted", False)),
                    truncated=bool(item.get("truncated", False)) or clipped,
                )
            )
        # Translate evaluator v1 fields at the boundary.
        legacy = (
            ("input_repr", "function_call", True),
            ("expected_repr", "expected_value", True),
            ("actual_repr", "actual_value", False),
            ("exception_type", "exception", False),
        )
        for field, observation_kind, trusted in legacy:
            if raw.get(field) is not None and not any(
                item.kind == observation_kind for item in observations
            ):
                value, clipped = sanitize_evidence_text(raw[field])
                observations.append(
                    EvidenceObservation(observation_kind, value, trusted, clipped)
                )
        count = raw.get("failed_test_count")
        if count is not None and not isinstance(count, int):
            raise AdapterError("evidence failed_test_count must be an integer")
        test_id = raw.get("test_id")
        if test_id is not None:
            if not isinstance(test_id, str):
                raise AdapterError("evidence test_id must be a string")
            test_id, test_clipped = sanitize_evidence_text(test_id, 256)
        else:
            test_clipped = False
        by_kind = {item.kind: item.value for item in observations}
        provenance = raw.get("provenance", f"{self.name}-evaluator")
        if not isinstance(provenance, str):
            raise AdapterError("evidence provenance must be a string")
        provenance, provenance_clipped = sanitize_evidence_text(provenance, 128)
        return FailureEvidence(
            kind=kind,
            message=message,
            benchmark=problem.benchmark,
            test_id=test_id,
            observations=tuple(observations),
            provenance=provenance,
            truncated=bool(raw.get("truncated", False)) or message_truncated or test_clipped
            or provenance_clipped or any(item.truncated for item in observations),
            input_repr=by_kind.get("function_call"),
            expected_repr=by_kind.get("expected_value"),
            actual_repr=by_kind.get("actual_value"),
            exception_type=by_kind.get("exception"),
            failed_test_count=count,
        )


_ADAPTERS: dict[str, BenchmarkAdapter] = {}


def register_benchmark_adapter(adapter: BenchmarkAdapter) -> None:
    if not adapter.name:
        raise ValueError("adapter name must not be empty")
    _ADAPTERS[adapter.name] = adapter


def get_benchmark_adapter(name: str) -> BenchmarkAdapter:
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        raise AdapterError(f"no evaluator adapter registered for benchmark: {name}") from exc


register_benchmark_adapter(PythonBenchmarkAdapter("humaneval-plus"))
register_benchmark_adapter(PythonBenchmarkAdapter("bigcodebench-hard"))
