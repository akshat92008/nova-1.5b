from nova_v12.patching import apply_operations, safe_resolve
from nova_v12.sandbox import SandboxRunner
from nova_v12.schemas import CommandSpec, FileSpec


def test_safe_resolve_blocks_escape(tmp_path):
    try:
        safe_resolve(tmp_path, "../secret")
    except ValueError:
        pass
    else:
        raise AssertionError("path escape was accepted")


def test_operations_are_atomic(tmp_path):
    (tmp_path / "a.txt").write_text("old")
    result = apply_operations(
        tmp_path,
        [
            {"action": "replace", "path": "a.txt", "search": "old", "replace": "new"},
            {"action": "replace", "path": "missing.txt", "search": "x", "replace": "y"},
        ],
    )
    assert not result.ok
    assert (tmp_path / "a.txt").read_text() == "old"


def test_operations_reject_workspace_symlinks(tmp_path):
    (tmp_path / "target.py").write_text("VALUE = 1\n")
    (tmp_path / "alias.py").symlink_to(tmp_path / "target.py")
    result = apply_operations(
        tmp_path,
        [{"action": "write", "path": "target.py", "content": "VALUE = 2\n"}],
    )
    assert not result.ok
    assert "symlinks are forbidden" in result.error
    assert (tmp_path / "target.py").read_text() == "VALUE = 1\n"


def test_sandbox_runs_without_shell():
    runner = SandboxRunner(timeout_seconds=10)
    temp, root = runner.make_workspace([FileSpec("check.py", "assert 2 + 2 == 4\n")])
    try:
        result = runner.run(root, CommandSpec(["python", "check.py"], timeout_seconds=5))
        assert result.exit_code == 0
    finally:
        temp.cleanup()
