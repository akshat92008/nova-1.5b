"""Typed protocol and evidence objects for Nova V12."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .constants import ESCALATION_SCHEMA, EVIDENCE_SCHEMA, PATCH_SCHEMA


class ContractError(ValueError):
    """Raised when input violates the Nova V12 contract."""


@dataclass(frozen=True)
class FileOperation:
    path: str
    action: str
    content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"path": self.path, "action": self.action}
        if self.action != "delete":
            result["content"] = self.content
        return result


@dataclass(frozen=True)
class PatchResponse:
    summary: str
    files: tuple[FileOperation, ...]
    schema_version: str = PATCH_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "summary": self.summary,
            "files": [item.to_dict() for item in self.files],
        }


@dataclass(frozen=True)
class EscalationResponse:
    reason_code: str
    message: str
    schema_version: str = ESCALATION_SCHEMA

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class AtomicTask:
    task_id: str
    instruction: str
    allowed_files: tuple[str, ...]
    test_command: tuple[str, ...]
    context_files: tuple[str, ...] = ()
    language: str = "unknown"
    task_kind: str = "patch"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AtomicTask":
        task_id = value.get("task_id", value.get("id"))
        instruction = value.get("instruction")
        allowed_files = value.get("allowed_files")
        test_command = value.get("test_command")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ContractError("task_id must be a non-empty string")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ContractError(f"{task_id}: instruction must be a non-empty string")
        if not isinstance(allowed_files, list) or not allowed_files:
            raise ContractError(f"{task_id}: allowed_files must be a non-empty list")
        if not all(isinstance(path, str) and path for path in allowed_files):
            raise ContractError(f"{task_id}: allowed_files must contain strings")
        if len(set(allowed_files)) != len(allowed_files):
            raise ContractError(f"{task_id}: allowed_files contains duplicates")
        if not isinstance(test_command, list) or not test_command:
            raise ContractError(f"{task_id}: test_command must be a non-empty argv list")
        if not all(isinstance(part, str) and part for part in test_command):
            raise ContractError(f"{task_id}: test_command must contain non-empty strings")
        context_files = value.get("context_files", [])
        if not isinstance(context_files, list) or not all(
            isinstance(path, str) and path for path in context_files
        ):
            raise ContractError(f"{task_id}: context_files must be a string list")
        return cls(
            task_id=task_id,
            instruction=instruction,
            allowed_files=tuple(allowed_files),
            test_command=tuple(test_command),
            context_files=tuple(context_files),
            language=str(value.get("language", "unknown")),
            task_kind=str(value.get("task_kind", "patch")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "allowed_files": list(self.allowed_files),
            "context_files": list(self.context_files),
            "test_command": list(self.test_command),
            "language": self.language,
            "task_kind": self.task_kind,
        }


@dataclass(frozen=True)
class CommandEvidence:
    argv: tuple[str, ...]
    exit_code: int | None
    timed_out: bool
    duration_seconds: float
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return not self.timed_out and self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["argv"] = list(self.argv)
        value["passed"] = self.passed
        return value


@dataclass(frozen=True)
class AttemptEvidence:
    attempt: int
    protocol_valid: bool
    scope_valid: bool
    patch_applied: bool
    tests_passed: bool
    changed_files: tuple[str, ...] = ()
    changed_lines: int = 0
    raw_output: str = ""
    error: str | None = None
    command: CommandEvidence | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["changed_files"] = list(self.changed_files)
        if self.command:
            value["command"] = self.command.to_dict()
        return value


@dataclass(frozen=True)
class RunEvidence:
    task_id: str
    status: str
    attempts: tuple[AttemptEvidence, ...]
    committed: bool
    final_response: PatchResponse | EscalationResponse | None
    schema_version: str = EVIDENCE_SCHEMA
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "passed" and self.committed

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "status": self.status,
            "committed": self.committed,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "final_response": (
                self.final_response.to_dict() if self.final_response is not None else None
            ),
            "metadata": self.metadata,
        }
