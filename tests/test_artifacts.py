import json
from hashlib import sha256
from pathlib import Path

import pytest

from tokenos.artifacts import ArtifactError, EventStore
from tokenos.models import (
    AttemptRecord, EvaluationResult, FailureFeedback, ModelUsage, Patch, RunConfig
)


def config() -> RunConfig:
    return RunConfig(
        run_id="run",
        strategies=("no-memory",),
        task_ids=("HumanEval/0",),
        dataset_path="data.gz",
    )


def test_manifest_events_and_config_validation(tmp_path: Path) -> None:
    store = EventStore(tmp_path, "run", create=True)
    store.write_manifest(config())
    store.append("request_started", prompt="safe", attempt=1)
    assert store.events()[0]["type"] == "request_started"
    assert store.events()[0]["schema_version"] == 2
    assert store.read_manifest()["schema_version"] == 2
    store.validate_config(config())


def test_manifest_without_new_optional_config_field_remains_valid(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path, "run", create=True)
    values = config().public_dict()
    values.pop("dataset_name")
    fingerprint_values = dict(values)
    fingerprint_values.pop("run_id")
    fingerprint_values.pop("created_at")
    fingerprint = sha256(
        json.dumps(
            fingerprint_values, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    store.manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_fingerprint": fingerprint,
                "config": values,
            }
        ),
        encoding="utf-8",
    )
    store.validate_config(config())


def test_corrupt_event_is_reported(tmp_path: Path) -> None:
    store = EventStore(tmp_path, "run", create=True)
    store.events_path.write_text("{bad\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="line 1"):
        store.events()


def test_evaluator_failure_cost_is_restored_on_resume(tmp_path: Path) -> None:
    store = EventStore(tmp_path, "run", create=True)
    store.append("evaluation_failed", estimated_cost_usd=0.125)
    assert store.committed_cost() == 0.125


def test_v1_attempt_event_loads_as_v2_memory(tmp_path: Path) -> None:
    store = EventStore(tmp_path, "run", create=True)
    record = AttemptRecord(
        run_id="run", task_id="HumanEval/0", strategy="patchsearch", attempt=1,
        prompt="p", completion="c", solution="s", response_id=None, request_id=None,
        usage=ModelUsage(), evaluation=EvaluationResult(
            "fail", "fail", False, FailureFeedback(
                kind="wrong_answer", message="mismatch", input_repr="()",
                expected_repr="True", actual_repr="False",
            )
        ), model_latency_seconds=0, evaluation_latency_seconds=0,
        total_latency_seconds=0, estimated_cost_usd=0, created_at="now",
        patches=(Patch("HumanEval/0", "f", "assert f(*()) == True", 1, "wrong_answer"),),
    )
    payload = record.to_dict()
    payload["evaluation"]["feedback"] = payload["evaluation"].pop("evidence")
    payload.pop("memories")
    payload.pop("retrieved_memories")
    store.append("attempt_completed", record=payload)
    restored = store.completed_attempts()[0]
    assert restored.memories[0].content == "assert f(*()) == True"
    assert restored.memories[0].provenance == "legacy-v1"
