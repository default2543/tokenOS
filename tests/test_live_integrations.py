import os
from pathlib import Path

import pytest

from tokenos.config import config_from_environment
from tokenos.dataset import DEFAULT_DATASET_PATH, file_sha256, load_problems
from tokenos.evaluator import DockerEvaluator, docker_status
from tokenos.models import RunConfig
from tokenos.provider import AzureOpenAIProvider
from tokenos.strategies import build_prompt, parse_completion


@pytest.mark.azure
def test_live_azure_generation() -> None:
    if os.environ.get("TOKENOS_RUN_AZURE_TEST") != "1":
        pytest.skip("set TOKENOS_RUN_AZURE_TEST=1 to make a paid Azure request")
    problems = load_problems(DEFAULT_DATASET_PATH)
    config = config_from_environment(
        strategies=("no-memory",),
        task_ids=("HumanEval/0",),
        dataset_path=DEFAULT_DATASET_PATH,
        dataset_sha256=file_sha256(DEFAULT_DATASET_PATH),
        max_attempts=1,
        concurrency=1,
    )
    prompt = build_prompt("no-memory", problems["HumanEval/0"], 1, [])
    response = AzureOpenAIProvider(config).generate(prompt)
    assert parse_completion(response.output_text)


@pytest.mark.docker
def test_live_docker_evaluator_passes_reference_completion() -> None:
    status = docker_status("tokenos-evalplus:0.3.1")
    if not status["image"]:
        pytest.skip("TokenOS evaluator image is not available")
    problems = load_problems(DEFAULT_DATASET_PATH)
    problem = problems["HumanEval/0"]
    config = RunConfig(
        run_id="docker-test",
        strategies=("no-memory",),
        task_ids=(problem.task_id,),
        dataset_path=str(DEFAULT_DATASET_PATH),
    )
    solution = problem.prompt + "    for i, elem in enumerate(numbers):\n        for elem2 in numbers[i + 1:]:\n            if abs(elem - elem2) < threshold:\n                return True\n    return False\n"
    assert DockerEvaluator(config).evaluate(problem, solution).passed


@pytest.mark.docker
@pytest.mark.parametrize(
    ("completion", "failure_kind", "has_oracle_counterexample"),
    [
        ("    return (\n", "syntax_error", False),
        ("    raise RuntimeError('test failure')\n", "runtime_error", True),
        ("    while True:\n        pass\n", "timeout", True),
        (
            "    import socket\n"
            "    socket.create_connection(('example.com', 80), timeout=0.1)\n"
            "    return False\n",
            "runtime_error",
            True,
        ),
    ],
)
def test_live_docker_failure_isolation(
    completion: str, failure_kind: str, has_oracle_counterexample: bool
) -> None:
    if os.environ.get("TOKENOS_RUN_DOCKER_SECURITY_TESTS") != "1":
        pytest.skip("set TOKENOS_RUN_DOCKER_SECURITY_TESTS=1 for isolation tests")
    status = docker_status("tokenos-evalplus:0.3.1")
    if not status["image"]:
        pytest.skip("TokenOS evaluator image is not available")
    problem = load_problems(DEFAULT_DATASET_PATH)["HumanEval/0"]
    config = RunConfig(
        run_id="docker-security-test",
        strategies=("no-memory",),
        task_ids=(problem.task_id,),
        dataset_path=str(DEFAULT_DATASET_PATH),
        evaluator_timeout_seconds=20,
    )
    result = DockerEvaluator(config).evaluate(problem, problem.prompt + completion)
    assert not result.passed
    assert result.feedback is not None
    assert result.feedback.kind == failure_kind
    assert (result.feedback.input_repr is not None) is has_oracle_counterexample
    assert (result.feedback.expected_repr is not None) is has_oracle_counterexample
    if has_oracle_counterexample:
        assert result.feedback.input_repr is not None
        assert result.feedback.input_repr.startswith("(")
