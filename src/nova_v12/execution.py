"""Transactional patch application and evidence-producing test execution."""

from __future__ import annotations

import difflib
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .constants import DEFAULT_TIMEOUT_SECONDS
from .policy import (
    resolve_workspace_path,
    safe_environment,
    validate_response_scope,
)
from .schema import AtomicTask, CommandEvidence, ContractError, PatchResponse


def _resource_limiter(timeout_seconds: int):
    """Return a POSIX pre-exec limiter; no-op behaviour is handled by the caller."""

    def limit() -> None:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (timeout_seconds, timeout_seconds + 1))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        except (ImportError, OSError, ValueError):
            return

    return limit


def run_test_command(
    workspace: Path,
    argv: tuple[str, ...],
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> CommandEvidence:
    """Run planner-owned test argv without a shell and capture bounded output."""
    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(argv),
            cwd=workspace,
            env=safe_environment(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            preexec_fn=_resource_limiter(timeout_seconds) if os.name == "posix" else None,
        )
        return CommandEvidence(
            argv=argv,
            exit_code=result.returncode,
            timed_out=False,
            duration_seconds=time.perf_counter() - started,
            stdout=result.stdout[-20_000:],
            stderr=result.stderr[-20_000:],
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
        )
        stderr = (
            exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        )
        return CommandEvidence(
            argv=argv,
            exit_code=None,
            timed_out=True,
            duration_seconds=time.perf_counter() - started,
            stdout=(stdout or "")[-20_000:],
            stderr=(stderr or "")[-20_000:],
        )
    except OSError as exc:
        return CommandEvidence(
            argv=argv,
            exit_code=None,
            timed_out=False,
            duration_seconds=time.perf_counter() - started,
            stdout="",
            stderr=str(exc),
        )


def _changed_line_count(before: str, after: str) -> int:
    diff = difflib.ndiff(before.splitlines(), after.splitlines())
    return sum(1 for line in diff if line.startswith(("+ ", "- ")))


def apply_response(
    workspace: Path,
    task: AtomicTask,
    response: PatchResponse,
) -> tuple[tuple[str, ...], int]:
    """Apply a validated response and return changed paths and changed-line count."""
    changed = validate_response_scope(task, response, workspace)
    changed_lines = 0
    for operation in response.files:
        target = resolve_workspace_path(workspace, operation.path)
        before = target.read_text(encoding="utf-8") if target.exists() else ""
        if operation.action == "delete":
            target.unlink()
            after = ""
        else:
            assert operation.content is not None
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(operation.content, encoding="utf-8")
            after = operation.content
        changed_lines += _changed_line_count(before, after)
    return changed, changed_lines


def copy_workspace(source: Path, destination: Path) -> None:
    """Copy regular project files while excluding VCS and generated caches."""
    if not source.is_dir():
        raise ContractError(f"workspace is not a directory: {source}")
    for candidate in source.rglob("*"):
        if candidate.is_symlink():
            raise ContractError(
                f"source workspace contains a symlink: {candidate.relative_to(source)}"
            )

    def ignore(_directory: str, names: list[str]) -> set[str]:
        blocked = {
            ".git",
            ".hg",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".svn",
            ".venv",
            "__pycache__",
            "build",
            "dist",
            "node_modules",
        }
        return blocked.intersection(names)

    shutil.copytree(source, destination, symlinks=True, ignore=ignore)


def verify_response(
    workspace: Path,
    task: AtomicTask,
    response: PatchResponse,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[tuple[str, ...], int, CommandEvidence]:
    """Apply and test a response in an isolated temporary copy."""
    with tempfile.TemporaryDirectory(prefix="nova-v12-verify-") as temp:
        staged = Path(temp) / "workspace"
        copy_workspace(workspace, staged)
        changed, changed_lines = apply_response(staged, task, response)
        command = run_test_command(staged, task.test_command, timeout_seconds=timeout_seconds)
        return changed, changed_lines, command


def commit_verified_response(
    workspace: Path,
    task: AtomicTask,
    response: PatchResponse,
) -> tuple[tuple[str, ...], int]:
    """Commit a response already verified in a copy, with rollback on write failure."""
    validate_response_scope(task, response, workspace)
    backups: dict[str, bytes | None] = {}
    for operation in response.files:
        target = resolve_workspace_path(workspace, operation.path)
        backups[operation.path] = target.read_bytes() if target.exists() else None
    try:
        return apply_response(workspace, task, response)
    except Exception:
        for raw, content in backups.items():
            target = resolve_workspace_path(workspace, raw)
            if content is None:
                if target.exists() and target.is_file():
                    target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        raise
