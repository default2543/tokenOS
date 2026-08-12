from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import os
from pathlib import Path
import platform
import secrets
from typing import Mapping

from tokenos.models import RunConfig, StrategyName


class ConfigurationError(ValueError):
    pass


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(3)}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _required_float(env: Mapping[str, str], name: str) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        raise ConfigurationError(f"{name} is required")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be numeric") from exc
    if value < 0:
        raise ConfigurationError(f"{name} cannot be negative")
    return value


def config_from_environment(
    *,
    strategies: tuple[StrategyName, ...],
    task_ids: tuple[str, ...],
    dataset_path: Path,
    dataset_sha256: str,
    max_attempts: int = 5,
    concurrency: int = 4,
    budget_usd: float = 10.0,
    run_id: str | None = None,
    env: Mapping[str, str] | None = None,
) -> RunConfig:
    source = os.environ if env is None else env
    endpoint = source.get("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        raise ConfigurationError("AZURE_OPENAI_ENDPOINT is required")
    auth = source.get("AZURE_OPENAI_AUTH", "api_key").strip().lower()
    if auth not in {"api_key", "entra"}:
        raise ConfigurationError("AZURE_OPENAI_AUTH must be api_key or entra")
    if auth == "api_key" and not source.get("AZURE_OPENAI_API_KEY", "").strip():
        raise ConfigurationError("AZURE_OPENAI_API_KEY is required for api_key auth")
    if max_attempts < 1 or concurrency < 1 or budget_usd <= 0:
        raise ConfigurationError("attempts, concurrency, and budget must be positive")

    return RunConfig(
        run_id=run_id or new_run_id(),
        strategies=strategies,
        task_ids=task_ids,
        dataset_path=str(dataset_path),
        dataset_sha256=dataset_sha256,
        azure_endpoint=endpoint,
        azure_deployment=source.get(
            "AZURE_OPENAI_DEPLOYMENT", "tokenos-gpt-54-mini"
        ).strip(),
        azure_auth=auth,  # type: ignore[arg-type]
        max_attempts=max_attempts,
        concurrency=concurrency,
        budget_usd=budget_usd,
        input_usd_per_mtok=_required_float(
            source, "AZURE_OPENAI_INPUT_USD_PER_MTOK"
        ),
        cached_input_usd_per_mtok=_required_float(
            source, "AZURE_OPENAI_CACHED_INPUT_USD_PER_MTOK"
        ),
        output_usd_per_mtok=_required_float(
            source, "AZURE_OPENAI_OUTPUT_USD_PER_MTOK"
        ),
        created_at=utc_now(),
        metadata={
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    )


def config_from_manifest(
    manifest: dict,
    *,
    env: Mapping[str, str] | None = None,
    budget_usd: float | None = None,
) -> RunConfig:
    config_data = dict(manifest["config"])
    config_data["strategies"] = tuple(config_data["strategies"])
    config_data["task_ids"] = tuple(config_data["task_ids"])
    config = RunConfig(**config_data)
    source = os.environ if env is None else env
    auth = source.get("AZURE_OPENAI_AUTH", config.azure_auth).strip().lower()
    if auth not in {"api_key", "entra"}:
        raise ConfigurationError("AZURE_OPENAI_AUTH must be api_key or entra")
    if auth == "api_key" and not source.get("AZURE_OPENAI_API_KEY", "").strip():
        raise ConfigurationError("AZURE_OPENAI_API_KEY is required for api_key auth")
    return replace(
        config,
        azure_auth=auth,  # type: ignore[arg-type]
        budget_usd=config.budget_usd if budget_usd is None else budget_usd,
    )
