from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any, Literal

StrategyName = Literal[
    "no-memory",
    "full-history",
    "patchsearch",
    "patchsearch-explained",
    "patchsearch-crosstask",
    "patchsearch-with-code",
]
FailureKind = Literal[
    "wrong_answer",
    "runtime_error",
    "syntax_error",
    "missing_entry_point",
    "timeout",
    "invalid_output",
    "infrastructure_error",
]
EvidenceObservationKind = Literal[
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
]
MemoryKind = Literal["executable_constraint", "failure_summary"]


@dataclass(frozen=True, slots=True)
class Problem:
    task_id: str
    prompt: str
    entry_point: str
    benchmark: str = "humaneval-plus"
    test_code: str | None = None
    libraries: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class EvidenceObservation:
    kind: EvidenceObservationKind
    value: str
    trusted: bool = False
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class FailureEvidence:
    kind: FailureKind
    message: str
    benchmark: str = ""
    test_id: str | None = None
    observations: tuple[EvidenceObservation, ...] = ()
    provenance: str = "evaluator"
    truncated: bool = False
    # Legacy v1 fields. Adapters translate these into typed observations.
    input_repr: str | None = None
    expected_repr: str | None = None
    actual_repr: str | None = None
    exception_type: str | None = None
    failed_test_count: int | None = None


# Source compatibility for callers that constructed v1 feedback directly.
FailureFeedback = FailureEvidence


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    base_status: str
    plus_status: str
    passed: bool
    feedback: FailureEvidence | None = None


@dataclass(frozen=True, slots=True)
class FailureMemory:
    """A bounded, benchmark-neutral memory derived from failure evidence."""

    task_id: str
    kind: MemoryKind
    content: str
    source_attempt: int
    failure_kind: FailureKind
    fingerprint: str
    validated: bool = True
    provenance: str = "evaluator"
    explanation: str | None = None

    @property
    def assertion(self) -> str:
        """Compatibility accessor for code that previously consumed Patch."""
        return self.content


@dataclass(frozen=True, slots=True)
class Patch:
    """A compact executable constraint learned from one failed attempt."""

    task_id: str
    entry_point: str
    assertion: str
    source_attempt: int
    failure_kind: FailureKind
    validated: bool = True
    explanation: str | None = None


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    run_id: str
    task_id: str
    strategy: StrategyName
    attempt: int
    prompt: str
    completion: str
    solution: str
    response_id: str | None
    request_id: str | None
    usage: ModelUsage
    evaluation: EvaluationResult
    model_latency_seconds: float
    evaluation_latency_seconds: float
    total_latency_seconds: float
    estimated_cost_usd: float
    created_at: str
    resolved_model: str | None = None
    patches: tuple[Patch, ...] = ()
    memories: tuple[FailureMemory, ...] = ()
    retrieved_memories: tuple[FailureMemory, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        evaluation = data["evaluation"]
        feedback = evaluation.pop("feedback", None)
        if feedback is not None:
            for legacy_field in (
                "input_repr", "expected_repr", "actual_repr", "exception_type"
            ):
                feedback.pop(legacy_field, None)
            evaluation["evidence"] = feedback
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttemptRecord":
        values = dict(data)
        values["usage"] = ModelUsage(**values["usage"])
        evaluation = dict(values["evaluation"])
        raw_evidence = evaluation.pop("evidence", evaluation.get("feedback"))
        evaluation.pop("feedback", None)
        if raw_evidence is not None:
            feedback = dict(raw_evidence)
            feedback["observations"] = tuple(
                EvidenceObservation(**item)
                for item in feedback.get("observations", ())
            )
            evaluation["feedback"] = FailureEvidence(**feedback)
        values["evaluation"] = EvaluationResult(**evaluation)
        values["patches"] = tuple(Patch(**item) for item in values.get("patches", ()))
        values["memories"] = tuple(
            FailureMemory(**item) for item in values.get("memories", ())
        )
        values["retrieved_memories"] = tuple(
            FailureMemory(**item) for item in values.get("retrieved_memories", ())
        )
        if not values["memories"] and values["patches"]:
            values["memories"] = tuple(
                FailureMemory(
                    task_id=patch.task_id,
                    kind="executable_constraint",
                    content=patch.assertion,
                    source_attempt=patch.source_attempt,
                    failure_kind=patch.failure_kind,
                    fingerprint=sha256(patch.assertion.encode()).hexdigest(),
                    validated=patch.validated,
                    provenance="legacy-v1",
                    explanation=patch.explanation,
                )
                for patch in values["patches"]
            )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class RunConfig:
    run_id: str
    strategies: tuple[StrategyName, ...]
    task_ids: tuple[str, ...]
    dataset_path: str
    dataset_name: str = "humaneval-plus"
    dataset_version: str = "v0.1.10"
    dataset_sha256: str = ""
    model_name: str = "gpt-5.4-mini"
    model_version: str = "2026-03-17"
    azure_endpoint: str = ""
    azure_deployment: str = "tokenos-gpt-54-mini"
    azure_auth: Literal["api_key", "entra"] = "api_key"
    reasoning_effort: str = "medium"
    max_output_tokens: int = 4096
    max_attempts: int = 5
    concurrency: int = 4
    budget_usd: float = 10.0
    input_usd_per_mtok: float = 0.0
    cached_input_usd_per_mtok: float = 0.0
    output_usd_per_mtok: float = 0.0
    evaluator_image: str = "tokenos-evalplus:0.3.1"
    evaluator_timeout_seconds: float = 90.0
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        data = self.public_dict()
        data.pop("run_id", None)
        data.pop("created_at", None)
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")
