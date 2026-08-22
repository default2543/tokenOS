import gzip
import json
from pathlib import Path

import pytest

from tokenos.dataset import DatasetError, load_problems, select_problems


def write_dataset(path: Path) -> None:
    rows = [
        {"task_id": "HumanEval/0", "prompt": "def a():\n", "entry_point": "a"},
        {"task_id": "HumanEval/1", "prompt": "def b():\n", "entry_point": "b"},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_load_and_select_problems(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl.gz"
    write_dataset(path)
    problems = load_problems(path)
    assert list(problems) == ["HumanEval/0", "HumanEval/1"]
    assert select_problems(problems, ["HumanEval/1"])[0].entry_point == "b"


def test_unknown_task_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl.gz"
    write_dataset(path)
    with pytest.raises(DatasetError, match="unknown task"):
        select_problems(load_problems(path), ["HumanEval/9"])


def test_load_bigcodebench_hard_problem(tmp_path: Path) -> None:
    path = tmp_path / "hard.jsonl.gz"
    row = {
        "task_id": "BigCodeBench/13",
        "complete_prompt": "import os\n\ndef task_func():\n",
        "entry_point": "task_func",
        "test": "import unittest\nclass TestCases(unittest.TestCase):\n    pass\n",
        "libs": "['os']",
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")

    problem = load_problems(path, dataset_name="bigcodebench-hard")[
        "BigCodeBench/13"
    ]
    assert problem.benchmark == "bigcodebench-hard"
    assert problem.prompt == row["complete_prompt"]
    assert problem.test_code == row["test"]


def test_unknown_dataset_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl.gz"
    write_dataset(path)
    with pytest.raises(DatasetError, match="unknown dataset"):
        load_problems(path, dataset_name="other")
