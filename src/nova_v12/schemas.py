from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

TaskCategory = Literal[
    "code_generation",
    "debugging",
    "repository_editing",
    "tool_use",
    "fim",
    "instruction_following",
]


@dataclass(slots=True)
class CommandSpec:
    command: list[str]
    timeout_seconds: int = 30
    expected_exit_code: int = 0
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CommandSpec":
        command = value.get("command")
        if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
            raise ValueError("command must be a list of strings; shell strings are not accepted")
        return cls(
            command=command,
            timeout_seconds=int(value.get("timeout_seconds", 30)),
            expected_exit_code=int(value.get("expected_exit_code", 0)),
            env={str(k): str(v) for k, v in value.get("env", {}).items()},
        )


@dataclass(slots=True)
class FileSpec:
    path: str
    content: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FileSpec":
        path = str(value["path"])
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError(f"unsafe task file path: {path}")
        return cls(path=path, content=str(value.get("content", "")))


@dataclass(slots=True)
class ConstraintSpec:
    kind: str
    value: Any = None
    negate: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConstraintSpec":
        return cls(
            kind=str(value["kind"]),
            value=value.get("value"),
            negate=bool(value.get("negate", False)),
        )


@dataclass(slots=True)
class EvalTask:
    id: str
    category: TaskCategory
    prompt: str
    language: str = "text"
    files: list[FileSpec] = field(default_factory=list)
    tests: list[CommandSpec] = field(default_factory=list)
    constraints: list[ConstraintSpec] = field(default_factory=list)
    expected_files_modified: list[str] = field(default_factory=list)
    prefix: str = ""
    suffix: str = ""
    reference: str | None = None
    tool_budget: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvalTask":
        category = value.get("category")
        if category not in {
            "code_generation",
            "debugging",
            "repository_editing",
            "tool_use",
            "fim",
            "instruction_following",
        }:
            raise ValueError(f"unsupported category: {category}")
        return cls(
            id=str(value["id"]),
            category=category,
            prompt=str(value.get("prompt", "")),
            language=str(value.get("language", "text")).lower(),
            files=[FileSpec.from_dict(x) for x in value.get("files", [])],
            tests=[CommandSpec.from_dict(x) for x in value.get("tests", [])],
            constraints=[ConstraintSpec.from_dict(x) for x in value.get("constraints", [])],
            expected_files_modified=[str(x) for x in value.get("expected_files_modified", [])],
            prefix=str(value.get("prefix", "")),
            suffix=str(value.get("suffix", "")),
            reference=value.get("reference"),
            tool_budget=int(value.get("tool_budget", 0)),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenerationResult:
    candidate_id: str
    task_id: str
    category: str
    raw_output: str
    latency_seconds: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    model_revision: str | None = None
    backend: str | None = None
    error: str | None = None
    task: dict[str, Any] | None = None
    interactions: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationResult":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CheckResult:
    name: str
    passed: bool
    weight: float = 1.0
    details: str = ""


@dataclass(slots=True)
class TaskScore:
    candidate_id: str
    task_id: str
    category: str
    score: float
    checks: list[CheckResult]
    execution: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CodeRecord:
    source: str
    repository: str
    revision: str
    path: str
    licence: str
    language: str
    content: str
    source_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        normalised = self.content.replace("\r\n", "\n").rstrip() + "\n"
        return hashlib.sha256(normalised.encode("utf-8", errors="replace")).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CodeRecord":
        return cls(
            source=str(value.get("source", "unknown")),
            repository=str(value.get("repository", value.get("repo_name", "unknown"))),
            revision=str(value.get("revision", value.get("commit", value.get("revision_id", "")))),
            path=str(value.get("path", "")),
            licence=str(value.get("licence", value.get("license", ""))),
            language=str(value.get("language", "unknown")).lower(),
            content=str(value.get("content", "")),
            source_id=str(value.get("source_id", value.get("blob_id", ""))),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["content_hash"] = self.content_hash
        return payload


def load_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL object required at {path}:{line_number}")
            yield value


def write_jsonl(path: str | Path, values: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
