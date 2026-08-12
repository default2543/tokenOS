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

