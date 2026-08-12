from __future__ import annotations

from dataclasses import dataclass
import json
import os
import random
import time
from typing import Any, Callable

from tokenos.models import ModelUsage, RunConfig
from tokenos.strategies import SYSTEM_INSTRUCTIONS


class ModelProviderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelResponse:
    output_text: str
    response_id: str | None
    request_id: str | None
    usage: ModelUsage
    latency_seconds: float
    resolved_model: str | None = None


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _usage(response: Any) -> ModelUsage:
    usage = _value(response, "usage", {})
    input_details = _value(usage, "input_tokens_details", {})
    output_details = _value(usage, "output_tokens_details", {})
    return ModelUsage(
        input_tokens=int(_value(usage, "input_tokens", 0) or 0),
        cached_input_tokens=int(_value(input_details, "cached_tokens", 0) or 0),
        output_tokens=int(_value(usage, "output_tokens", 0) or 0),
        reasoning_tokens=int(_value(output_details, "reasoning_tokens", 0) or 0),
    )


class AzureOpenAIProvider:
    def __init__(
        self,
        config: RunConfig,
        *,
        client: Any | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._sleep = sleeper
        self.client = client if client is not None else self._make_client(config)

    @staticmethod
    def _make_client(config: RunConfig) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelProviderError(
                "openai is not installed; install the TokenOS package first"
            ) from exc

        base_url = f"{config.azure_endpoint}/openai/v1/"
        if config.azure_auth == "api_key":
            api_key: str | Callable[[], str] = os.environ.get(
                "AZURE_OPENAI_API_KEY", ""
            )
        else:
            try:
                from azure.identity import (
                    DefaultAzureCredential,
                    get_bearer_token_provider,
                )
            except ImportError as exc:
                raise ModelProviderError(
                    "azure-identity is required for Entra ID authentication"
                ) from exc
            api_key = get_bearer_token_provider(
                DefaultAzureCredential(), "https://ai.azure.com/.default"
            )
        return OpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    def generate(self, prompt: str) -> ModelResponse:
        started = time.monotonic()
        last_error: BaseException | None = None
        for retry in range(4):
            try:
                response = self.client.responses.create(
                    model=self.config.azure_deployment,
                    instructions=SYSTEM_INSTRUCTIONS,
                    input=prompt,
                    reasoning={"effort": self.config.reasoning_effort},
                    max_output_tokens=self.config.max_output_tokens,
                    store=False,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "coding_completion",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {"completion": {"type": "string"}},
                                "required": ["completion"],
                                "additionalProperties": False,
                            },
                        }
                    },
                )
                return ModelResponse(
                    output_text=str(_value(response, "output_text", "")),
                    response_id=_value(response, "id"),
                    request_id=_value(response, "_request_id"),
                    usage=_usage(response),
                    latency_seconds=time.monotonic() - started,
                    resolved_model=_value(response, "model"),
                )
            except Exception as exc:
                last_error = exc
                status = _value(exc, "status_code")
                retryable = status == 429 or (
                    isinstance(status, int) and 500 <= status < 600
                )
                retryable = retryable or isinstance(
                    exc, (ConnectionError, TimeoutError)
                )
                if not retryable or retry == 3:
                    break
                retry_after = _retry_after_seconds(exc)
                delay = retry_after if retry_after is not None else 2**retry
                self._sleep(delay + random.uniform(0, min(0.25, delay / 10)))
        assert last_error is not None
        raise ModelProviderError(
            f"Azure request failed after retries: {type(last_error).__name__}: {last_error}"
        ) from last_error


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = _value(exc, "response")
    headers = _value(response, "headers", {})
    raw = _value(headers, "retry-after")
    if raw is None and hasattr(headers, "get"):
        raw = headers.get("retry-after")
    try:
        return max(0.0, float(raw)) if raw is not None else None
    except (TypeError, ValueError):
        return None


def structured_output_overhead_bytes() -> int:
    schema = {
        "instructions": SYSTEM_INSTRUCTIONS,
        "format": {
            "type": "json_schema",
            "name": "coding_completion",
            "schema": {"completion": "string"},
        },
    }
    return len(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
