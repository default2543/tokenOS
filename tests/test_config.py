from pathlib import Path

import pytest

from tokenos.config import ConfigurationError, config_from_environment


BASE_ENV = {
    "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
    "AZURE_OPENAI_DEPLOYMENT": "tokenos-gpt-54-mini",
    "AZURE_OPENAI_INPUT_USD_PER_MTOK": "0.75",
    "AZURE_OPENAI_CACHED_INPUT_USD_PER_MTOK": "0.075",
    "AZURE_OPENAI_OUTPUT_USD_PER_MTOK": "4.5",
}


def make_config(env: dict[str, str]):
    return config_from_environment(
        strategies=("no-memory",),
        task_ids=("HumanEval/0",),
        dataset_path=Path("dataset.gz"),
        dataset_sha256="abc",
        env=env,
        run_id="run-1",
    )


def test_api_key_auth_requires_key() -> None:
    env = {**BASE_ENV, "AZURE_OPENAI_AUTH": "api_key"}
    with pytest.raises(ConfigurationError, match="API_KEY"):
        make_config(env)


def test_api_key_auth_configuration() -> None:
    config = make_config(
        {
            **BASE_ENV,
            "AZURE_OPENAI_AUTH": "api_key",
            "AZURE_OPENAI_API_KEY": "secret",
        }
    )
    assert config.azure_auth == "api_key"
    assert config.azure_endpoint == "https://example.openai.azure.com"
    assert "secret" not in str(config.public_dict())


def test_entra_auth_does_not_require_api_key() -> None:
    config = make_config({**BASE_ENV, "AZURE_OPENAI_AUTH": "entra"})
    assert config.azure_auth == "entra"


def test_rates_are_required() -> None:
    env = {
        **BASE_ENV,
        "AZURE_OPENAI_AUTH": "entra",
        "AZURE_OPENAI_OUTPUT_USD_PER_MTOK": "",
    }
    with pytest.raises(ConfigurationError, match="OUTPUT_USD"):
        make_config(env)



def test_provider_api_defaults_to_responses() -> None:
    config = make_config({**BASE_ENV, "AZURE_OPENAI_API_KEY": "k"})
    assert config.provider_api == "responses"
    assert config.model_name == "gpt-5.4-mini"


def test_chat_provider_records_the_model_it_runs() -> None:
    config = make_config(
        {
            **BASE_ENV,
            "AZURE_OPENAI_API_KEY": "k",
            "TOKENOS_PROVIDER_API": "chat",
            "TOKENOS_MODEL_NAME": "qwen3-coder-30b-a3b",
            "TOKENOS_MODEL_VERSION": "2507",
        }
    )
    assert config.provider_api == "chat"
    assert config.model_name == "qwen3-coder-30b-a3b"
    assert config.model_version == "2507"


def test_unknown_provider_api_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="TOKENOS_PROVIDER_API"):
        make_config(
            {**BASE_ENV, "AZURE_OPENAI_API_KEY": "k",
             "TOKENOS_PROVIDER_API": "completions"}
        )
