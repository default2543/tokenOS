import json
from hashlib import sha256
from pathlib import Path

import pytest

from tokenos.artifacts import ArtifactError, EventStore
from tokenos.models import RunConfig


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
