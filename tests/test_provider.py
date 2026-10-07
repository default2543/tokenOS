from dataclasses import replace
from types import SimpleNamespace

from tokenos.models import RunConfig
from tokenos.provider import AzureOpenAIProvider, base_url


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


def test_sdk_connection_errors_are_retried_not_fatal() -> None:
    # openai.APIConnectionError derives from the SDK's base class, not the
    # builtins, and exposes no status code. Treating it as fatal ended a whole
    # trajectory on any transient network blip -- a sleeping laptop included.
    import openai

    from tokenos.provider import _retryable_transport_errors

    assert issubclass(openai.APIConnectionError, _retryable_transport_errors())
    assert not issubclass(openai.APIConnectionError, (ConnectionError, TimeoutError))

    class Dropping:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise openai.APIConnectionError(request=SimpleNamespace())
            return SimpleNamespace(
                output_text='{"completion":"    return 1\\n"}',
                id="resp",
                usage=None,
                model="m",
            )

    responses = Dropping()
    config = RunConfig(
        run_id="run",
        strategies=("no-memory",),
        task_ids=("HumanEval/0",),
        dataset_path="data.gz",
        azure_endpoint="https://example",
    )
    sleeps: list[float] = []
    AzureOpenAIProvider(
        config, client=SimpleNamespace(responses=responses), sleeper=sleeps.append
    ).generate("prompt")
    assert responses.calls == 3
    assert len(sleeps) == 2


class ChatCompletions:
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
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"completion":"    return True"}'
                    )
                )
            ],
            id="chat-1",
            model="qwen3-coder",
            _request_id="req-1",
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                prompt_tokens_details=SimpleNamespace(cached_tokens=10),
                completion_tokens_details=None,
            ),
        )


def _chat_config() -> RunConfig:
    return RunConfig(
        run_id="run",
        strategies=("no-memory",),
        task_ids=("HumanEval/0",),
        dataset_path="data.gz",
        azure_endpoint="https://example",
        azure_deployment="qwen3-coder",
        provider_api="chat",
    )


def test_chat_provider_sends_structured_chat_request() -> None:
    chat = ChatCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=chat))
    result = AzureOpenAIProvider(_chat_config(), client=client).generate(
        "prompt", instructions="be terse"
    )
    call = chat.calls[0]
    assert call["model"] == "qwen3-coder"
    assert call["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "prompt"},
    ]
    assert call["response_format"]["json_schema"]["strict"] is True
    assert "reasoning" not in call
    assert result.output_text == '{"completion":"    return True"}'
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 20
    assert result.usage.cached_input_tokens == 10
    assert result.usage.reasoning_tokens == 0
    assert result.resolved_model == "qwen3-coder"


def test_chat_provider_retries_throttling() -> None:
    chat = ChatCompletions(failures=1)
    client = SimpleNamespace(chat=SimpleNamespace(completions=chat))
    AzureOpenAIProvider(_chat_config(), client=client, sleeper=lambda _: None).generate(
        "prompt"
    )
    assert len(chat.calls) == 2


def test_base_url_keeps_explicit_v1_endpoints() -> None:
    assert base_url("https://res.openai.azure.com/") == (
        "https://res.openai.azure.com/openai/v1/"
    )
    assert base_url("https://openrouter.ai/api/v1") == "https://openrouter.ai/api/v1/"
    assert base_url("https://res.services.ai.azure.com/openai/v1/") == (
        "https://res.services.ai.azure.com/openai/v1/"
    )
