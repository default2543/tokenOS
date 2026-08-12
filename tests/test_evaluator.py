import json
from types import SimpleNamespace
from unittest.mock import patch

from tokenos.evaluator import DockerEvaluator
from tokenos.models import Problem, RunConfig


def test_docker_evaluator_uses_locked_down_container() -> None:
    config = RunConfig(
        run_id="run",
        strategies=("no-memory",),
        task_ids=("HumanEval/0",),
        dataset_path="data.gz",
    )
    payload = {
        "base_status": "fail",
        "plus_status": "fail",
        "passed": False,
        "feedback": {"kind": "wrong_answer", "message": "mismatch"},
    }
    completed = SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
    with patch("tokenos.evaluator.subprocess.run", return_value=completed) as run:
        result = DockerEvaluator(config).evaluate(
            Problem("HumanEval/0", "def f():\n", "f"), "def f():\n return 0"
        )
    command = run.call_args.args[0]
    assert "none" in command
    assert "--read-only" in command
    assert "no-new-privileges" in command
    assert "--cap-drop" in command
    assert result.feedback is not None
    assert result.feedback.kind == "wrong_answer"

