from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from tokenos.models import EvaluationResult, FailureFeedback, Problem, RunConfig


class EvaluatorError(RuntimeError):
    pass


class DockerEvaluator:
    def __init__(self, config: RunConfig) -> None:
        self.config = config

    def evaluate(self, problem: Problem, solution: str) -> EvaluationResult:
        job = json.dumps({"task_id": problem.task_id, "solution": solution})
        command = [
            "docker",
            "run",
            "--rm",
            "--interactive",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "512m",
            "--cpus",
            "1",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            self.config.evaluator_image,
        ]
        try:
            completed = subprocess.run(
                command,
                input=job,
                capture_output=True,
                text=True,
                timeout=self.config.evaluator_timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise EvaluatorError("Docker CLI is not installed") from exc
        except subprocess.TimeoutExpired as exc:
            raise EvaluatorError("evaluator container exceeded the host timeout") from exc
        if completed.returncode != 0:
            error = (completed.stderr.strip() or completed.stdout.strip())[-2000:]
            raise EvaluatorError(
                f"evaluator container exited with {completed.returncode}: {error}"
            )
        try:
            payload: dict[str, Any] = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise EvaluatorError("evaluator returned invalid JSON") from exc
        feedback = payload.get("feedback")
        return EvaluationResult(
            base_status=payload["base_status"],
            plus_status=payload["plus_status"],
            passed=bool(payload["passed"]),
            feedback=FailureFeedback(**feedback) if feedback is not None else None,
        )


def docker_status(image: str) -> dict[str, bool]:
    cli = shutil.which("docker") is not None
    if not cli:
        return {"cli": False, "daemon": False, "image": False}
    daemon = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, check=False
    ).returncode == 0
    image_exists = daemon and subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    ).returncode == 0
    return {"cli": cli, "daemon": daemon, "image": image_exists}
