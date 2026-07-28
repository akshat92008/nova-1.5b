from __future__ import annotations

import os

try:
    import resource
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .schemas import CommandSpec, FileSpec


@dataclass(slots=True)
class CommandResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class VerificationResult:
    passed: bool
    commands: list[CommandResult]
    changed_files: list[str]
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "commands": [x.to_dict() for x in self.commands],
            "changed_files": self.changed_files,
            "error": self.error,
        }


class SandboxRunner:
    """Local process sandbox with workspace confinement and resource limits.

    This is suitable for trusted benchmark fixtures. For untrusted public code, run this
    package inside a container/VM with network disabled. The runner never invokes a shell.
    """

    def __init__(
        self,
        timeout_seconds: int = 30,
        memory_mb: int = 1024,
        max_output_chars: int = 200_000,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.memory_mb = memory_mb
        self.max_output_chars = max_output_chars

    def make_workspace(self, files: Iterable[FileSpec]) -> tuple[tempfile.TemporaryDirectory, Path]:
        temp = tempfile.TemporaryDirectory(prefix="nova-eval-")
        root = Path(temp.name)
        for file_spec in files:
            target = (root / file_spec.path).resolve()
            if root.resolve() not in target.parents:
                temp.cleanup()
                raise ValueError(f"fixture path escapes workspace: {file_spec.path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(file_spec.content, encoding="utf-8")
        return temp, root

    def run(self, root: Path, spec: CommandSpec) -> CommandResult:
        command = list(spec.command)
        if not command:
            raise ValueError("empty command")
        executable = shutil.which(command[0])
        if executable is None and command[0] == "python":
            import sys

            executable = shutil.which("python3") or sys.executable
        if executable is None:
            return CommandResult(command, 127, "", f"executable not found: {command[0]}", 0.0)
        command[0] = executable

        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(root),
            "TMPDIR": str(root / ".tmp"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            **spec.env,
        }
        (root / ".tmp").mkdir(exist_ok=True)

        def limits() -> None:
            if resource is None:
                return
            try:
                memory = self.memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (max(1, spec.timeout_seconds), max(2, spec.timeout_seconds + 1)),
                )
                resource.setrlimit(
                    resource.RLIMIT_FSIZE,
                    (64 * 1024 * 1024, 64 * 1024 * 1024),
                )
                resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
            except (ValueError, OSError):
                pass

        started = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=min(spec.timeout_seconds, self.timeout_seconds),
                check=False,
                preexec_fn=limits if os.name == "posix" else None,
            )
            duration = time.perf_counter() - started
            return CommandResult(
                command=spec.command,
                exit_code=proc.returncode,
                stdout=proc.stdout[-self.max_output_chars :],
                stderr=proc.stderr[-self.max_output_chars :],
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - started
            return CommandResult(
                command=spec.command,
                exit_code=124,
                stdout=(exc.stdout or "")[-self.max_output_chars :]
                if isinstance(exc.stdout, str)
                else "",
                stderr=(exc.stderr or "")[-self.max_output_chars :]
                if isinstance(exc.stderr, str)
                else "",
                duration_seconds=duration,
                timed_out=True,
            )

    def verify(
        self, root: Path, tests: Iterable[CommandSpec], changed_files: list[str]
    ) -> VerificationResult:
        results: list[CommandResult] = []
        for test in tests:
            result = self.run(root, test)
            results.append(result)
            if result.exit_code != test.expected_exit_code:
                return VerificationResult(
                    False, results, changed_files, "verification command failed"
                )
        return VerificationResult(True, results, changed_files)
