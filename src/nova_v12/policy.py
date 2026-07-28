"""Fail-closed filesystem, scope, and command policy."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from .constants import MAX_CHANGED_FILES, MAX_FILE_BYTES
from .schema import AtomicTask, ContractError, PatchResponse

BLOCKED_PATH_PARTS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".env",
        ".ssh",
        "__pycache__",
        "node_modules",
    }
)

ALLOWED_TEST_EXECUTABLES = frozenset(
    {
        "bash",
        "bun",
        "cargo",
        "go",
        "java",
        "mvn",
        "node",
        "npm",
        "npx",
        "pnpm",
        "poetry",
        "pytest",
        "python",
        "python3",
        "ruby",
        "uv",
    }
)


def normalise_relative_path(raw: str) -> str:
    """Return a canonical POSIX relative path or raise."""
    if not raw or "\x00" in raw or "\\" in raw:
        raise ContractError(f"unsafe path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or raw.startswith(("/", "~")):
        raise ContractError(f"path must be relative: {raw!r}")
    if any(part in ("", ".", "..") for part in path.parts):
        raise ContractError(f"path traversal is forbidden: {raw!r}")
    if any(part in BLOCKED_PATH_PARTS for part in path.parts):
        raise ContractError(f"blocked path: {raw!r}")
    return path.as_posix()


def resolve_workspace_path(workspace: Path, raw: str) -> Path:
    """Resolve a protocol path beneath workspace without following escapes."""
    normalised = normalise_relative_path(raw)
    root = workspace.resolve()
    candidate = root.joinpath(*PurePosixPath(normalised).parts)
    current = root
    for part in PurePosixPath(normalised).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"symlink parent is forbidden: {raw!r}")
    resolved_parent = candidate.parent.resolve()
    if resolved_parent != root and root not in resolved_parent.parents:
        raise ContractError(f"path escapes workspace: {raw!r}")
    if candidate.is_symlink():
        raise ContractError(f"symlink target is forbidden: {raw!r}")
    return candidate


def validate_task(task: AtomicTask) -> None:
    """Validate planner-owned task fields."""
    if len(task.allowed_files) > MAX_CHANGED_FILES:
        raise ContractError(
            f"atomic task allows {len(task.allowed_files)} files; maximum is {MAX_CHANGED_FILES}"
        )
    for path in (*task.allowed_files, *task.context_files):
        normalise_relative_path(path)
    executable = Path(task.test_command[0]).name
    if executable not in ALLOWED_TEST_EXECUTABLES:
        raise ContractError(f"test executable is not allowlisted: {executable}")
    forbidden_args = {"-c", "--eval", "-e"}
    if executable in {"bash", "python", "python3", "node", "ruby"} and any(
        arg in forbidden_args for arg in task.test_command[1:]
    ):
        raise ContractError("inline code in test commands is forbidden")


def validate_response_scope(
    task: AtomicTask,
    response: PatchResponse,
    workspace: Path,
) -> tuple[str, ...]:
    """Validate a model response against the planner-owned atomic scope."""
    validate_task(task)
    if not response.files:
        raise ContractError("patch response must change at least one file")
    if len(response.files) > MAX_CHANGED_FILES:
        raise ContractError(f"response changes more than {MAX_CHANGED_FILES} files")

    allowed = {normalise_relative_path(path) for path in task.allowed_files}
    changed: list[str] = []
    seen: set[str] = set()
    for operation in response.files:
        path = normalise_relative_path(operation.path)
        if path in seen:
            raise ContractError(f"duplicate file operation: {path}")
        if path not in allowed:
            raise ContractError(f"file is outside planner scope: {path}")
        seen.add(path)
        changed.append(path)

        target = resolve_workspace_path(workspace, path)
        exists = target.exists()
        if target.exists() and not target.is_file():
            raise ContractError(f"target is not a regular file: {path}")
        if operation.action == "create" and exists:
            raise ContractError(f"create target already exists: {path}")
        if operation.action in {"update", "delete"} and not exists:
            raise ContractError(f"{operation.action} target does not exist: {path}")
        if operation.action != "delete":
            assert operation.content is not None
            if len(operation.content.encode("utf-8")) > MAX_FILE_BYTES:
                raise ContractError(f"file exceeds {MAX_FILE_BYTES} bytes: {path}")
    return tuple(changed)


def safe_environment() -> dict[str, str]:
    """Create a deterministic low-privilege-ish environment for test commands."""
    keep = {
        "LANG",
        "LC_ALL",
        "PATH",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "WINDIR",
    }
    env = {key: value for key, value in os.environ.items() if key in keep}
    env.update(
        {
            "CI": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "NO_COLOR": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    for key in list(env):
        if key.lower().endswith("proxy"):
            env.pop(key, None)
    return env
