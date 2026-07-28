import json

from nova_v12.eval.scorer import score_result
from nova_v12.schemas import GenerationResult


def code_task():
    return {
        "id": "add-one",
        "category": "code_generation",
        "prompt": "write add_one",
        "language": "python",
        "files": [
            {
                "path": "check.py",
                "content": (
                    "from solution import add_one\n"
                    "assert add_one(1) == 2\n"
                    "assert add_one(-1) == 0\n"
                ),
            }
        ],
        "tests": [{"command": ["python", "check.py"], "timeout_seconds": 5}],
        "metadata": {"output_path": "solution.py"},
    }


def test_wrong_code_is_not_full_score():
    result = GenerationResult(
        "bad", "add-one", "code_generation", "def add_one(value):\n    return 0", task=code_task()
    )
    score = score_result(result)
    assert score.score < 0.5
    assert not next(item for item in score.checks if item.name == "tests_pass").passed


def test_correct_code_passes():
    result = GenerationResult(
        "good",
        "add-one",
        "code_generation",
        "def add_one(value):\n    return value + 1",
        task=code_task(),
    )
    score = score_result(result)
    assert score.score == 1.0


def test_structured_patch_executes():
    task = {
        "id": "fix",
        "category": "debugging",
        "prompt": "fix",
        "language": "python",
        "files": [
            {"path": "mod.py", "content": "def value():\n    return 1\n"},
            {"path": "check.py", "content": "from mod import value\nassert value() == 2\n"},
        ],
        "tests": [{"command": ["python", "check.py"], "timeout_seconds": 5}],
        "expected_files_modified": ["mod.py"],
    }
    output = json.dumps(
        {
            "status": "patch",
            "operations": [
                {
                    "action": "replace",
                    "path": "mod.py",
                    "search": "return 1",
                    "replace": "return 2",
                }
            ],
        }
    )
    score = score_result(GenerationResult("good", "fix", "debugging", output, task=task))
    assert score.score == 1.0
