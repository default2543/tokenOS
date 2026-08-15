from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any, Literal

StrategyName = Literal["no-memory", "full-history", "patchsearch"]
FailureKind = Literal[
    "wrong_answer",
    "runtime_error",
    "syntax_error",
    "missing_entry_point",
    "timeout",
    "invalid_output",
    "infrastructure_error",
]


@dataclass(frozen=True, slots=True)
class Problem:
    task_id: str
    prompt: str
    entry_point: str


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
class FailureFeedback:
    kind: FailureKind
    message: str
    input_repr: str | None = None
    expected_repr: str | None = None
    actual_repr: str | None = None
    exception_type: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    base_status: str
    plus_status: str
    passed: bool
    feedback: FailureFeedback | None = None


@dataclass(frozen=True, slots=True)
class Patch:
    """A compact executable constraint learned from one failed attempt."""

    task_id: str
    entry_point: str
    assertion: str
    source_attempt: int
    failure_kind: FailureKind
    validated: bool = True


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttemptRecord":
        values = dict(data)
        values["usage"] = ModelUsage(**values["usage"])
        evaluation = dict(values["evaluation"])
        if evaluation.get("feedback") is not None:
            evaluation["feedback"] = FailureFeedback(**evaluation["feedback"])
        values["evaluation"] = EvaluationResult(**evaluation)
        values["patches"] = tuple(Patch(**item) for item in values.get("patches", ()))
        return cls(**values)


@dataclass(frozen=True, slots=True)
class RunConfig:
    run_id: str
    strategies: tuple[StrategyName, ...]
    task_ids: tuple[str, ...]
    dataset_path: str
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
