from dataclasses import replace
from types import SimpleNamespace

from tokenos.models import RunConfig
from tokenos.provider import AzureOpenAIProvider


class Responses:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            self.failures -= 1
            error = RuntimeError("throttled")
            error.status_code = 429
            raise error
        return SimpleNamespace(
            output_text='{"completion":"    return True"}',
            id="resp-1",
            model="gpt-5.4-mini-2026-03-17",
            _request_id="req-1",
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                input_tokens_details=SimpleNamespace(cached_tokens=10),
                output_tokens_details=SimpleNamespace(reasoning_tokens=5),
            ),
        )


def test_provider_request_is_stateless_and_structured() -> None:
    responses = Responses()
    client = SimpleNamespace(responses=responses)
    config = RunConfig(
        run_id="run",
        strategies=("no-memory",),
        task_ids=("HumanEval/0",),
        dataset_path="data.gz",
        azure_endpoint="https://example",
    )
    result = AzureOpenAIProvider(config, client=client).generate("prompt")
    call = responses.calls[0]
    assert call["store"] is False
    assert call["reasoning"] == {"effort": "medium"}
    assert call["text"]["format"]["strict"] is True
    assert result.usage.reasoning_tokens == 5
    assert result.usage.cached_input_tokens == 10
    assert result.resolved_model == "gpt-5.4-mini-2026-03-17"


def test_provider_retries_throttling_without_logical_attempt() -> None:
    responses = Responses(failures=2)
    sleeps = []
    client = SimpleNamespace(responses=responses)
    config = RunConfig(
        run_id="run",
        strategies=("no-memory",),
        task_ids=("HumanEval/0",),
        dataset_path="data.gz",
        azure_endpoint="https://example",
    )
    AzureOpenAIProvider(config, client=client, sleeper=sleeps.append).generate("prompt")
    assert len(responses.calls) == 3
    assert len(sleeps) == 2
