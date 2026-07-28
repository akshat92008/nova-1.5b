from pathlib import Path

import pytest

from nova_v12.execution import apply_response, copy_workspace, run_test_command, verify_response
from nova_v12.policy import normalise_relative_path, validate_task
from nova_v12.schema import AtomicTask, ContractError, FileOperation, PatchResponse


def task(**overrides):
    value = {
        "task_id": "t1",
        "instruction": "Set value to two.",
        "allowed_files": ["value.py"],
        "test_command": ["python", "check.py"],
    }
    value.update(overrides)
    return AtomicTask.from_dict(value)


def response(content="VALUE = 2\n", action="update", path="value.py"):
    return PatchResponse(
        summary="Set the value",
        files=(FileOperation(path=path, action=action, content=content),),
    )


def make_workspace(root: Path):
    (root / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "check.py").write_text(
        "from value import VALUE\nassert VALUE == 2\n",
        encoding="utf-8",
    )


def test_normalise_relative_path():
    assert normalise_relative_path("src/a.py") == "src/a.py"
    with pytest.raises(ContractError):
        normalise_relative_path("../a.py")


def test_task_rejects_more_than_three_files():
    item = task(allowed_files=["a", "b", "c", "d"])
    with pytest.raises(ContractError, match="maximum"):
        validate_task(item)


def test_task_rejects_inline_python():
    item = task(test_command=["python", "-c", "print(1)"])
    with pytest.raises(ContractError, match="inline code"):
        validate_task(item)


def test_apply_and_verify(tmp_path):
    make_workspace(tmp_path)
    changed, lines, command = verify_response(tmp_path, task(), response())
    assert changed == ("value.py",)
    assert lines == 2
    assert command.passed
    assert (tmp_path / "value.py").read_text() == "VALUE = 1\n"


def test_apply_rejects_out_of_scope(tmp_path):
    make_workspace(tmp_path)
    with pytest.raises(ContractError, match="outside planner scope"):
        apply_response(tmp_path, task(), response(path="other.py", action="create"))


def test_apply_rejects_action_mismatch(tmp_path):
    make_workspace(tmp_path)
    with pytest.raises(ContractError, match="already exists"):
        apply_response(tmp_path, task(), response(action="create"))


def test_command_failure_is_evidence(tmp_path):
    (tmp_path / "check.py").write_text("raise SystemExit(7)\n", encoding="utf-8")
    evidence = run_test_command(tmp_path, ("python", "check.py"))
    assert evidence.exit_code == 7
    assert not evidence.passed


def test_copy_workspace_rejects_symlink(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "real.txt").write_text("safe", encoding="utf-8")
    (source / "link.txt").symlink_to(source / "real.txt")
    with pytest.raises(ContractError, match="symlink"):
        copy_workspace(source, tmp_path / "copy")


def test_command_timeout_is_evidence(tmp_path):
    (tmp_path / "slow.py").write_text(
        "import time\ntime.sleep(5)\n",
        encoding="utf-8",
    )
    evidence = run_test_command(
        tmp_path,
        ("python", "slow.py"),
        timeout_seconds=1,
    )
    assert evidence.timed_out
    assert evidence.exit_code is None
    assert not evidence.passed


def test_missing_command_is_evidence(tmp_path):
    evidence = run_test_command(tmp_path, ("definitely-not-a-real-command",))
    assert evidence.exit_code is None
    assert "No such file" in evidence.stderr


def test_copy_workspace_requires_directory(tmp_path):
    with pytest.raises(ContractError, match="not a directory"):
        copy_workspace(tmp_path / "missing", tmp_path / "copy")
