"""Strict parser for the Nova V12 JSON output protocol."""

from __future__ import annotations

import json
from typing import Any

from .constants import (
    ESCALATION_SCHEMA,
    MAX_RESPONSE_BYTES,
    PATCH_SCHEMA,
)
from .policy import normalise_relative_path
from .schema import (
    ContractError,
    EscalationResponse,
    FileOperation,
    PatchResponse,
)

ESCALATION_CODES = frozenset(
    {"ambiguous", "scope_too_large", "missing_context", "unsafe", "unsupported"}
)


def _require_exact_keys(
    value: dict[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    label: str,
) -> None:
    optional = optional or set()
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ContractError(f"{label} missing keys: {sorted(missing)}")
    if unknown:
        raise ContractError(f"{label} has unknown keys: {sorted(unknown)}")


def parse_response(raw: str) -> PatchResponse | EscalationResponse:
    """Parse one response. Extra prose, fences, and unknown fields are rejected."""
    if not isinstance(raw, str):
        raise ContractError("model response must be text")
    if len(raw.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ContractError("model response exceeds size limit")
    stripped = raw.strip()
    if not stripped:
        raise ContractError("model response is empty")
    if stripped.startswith("```") or stripped.endswith("```"):
        raise ContractError("Markdown fences are forbidden")
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ContractError("top-level response must be an object")

    schema = value.get("schema_version")
    if schema == PATCH_SCHEMA:
        _require_exact_keys(
            value,
            required={"schema_version", "summary", "files"},
            label="patch response",
        )
        summary = value["summary"]
        files = value["files"]
        if not isinstance(summary, str) or not summary.strip() or len(summary) > 240:
            raise ContractError("summary must be 1-240 characters")
        if not isinstance(files, list):
            raise ContractError("files must be an array")
        operations: list[FileOperation] = []
        for index, item in enumerate(files):
            if not isinstance(item, dict):
                raise ContractError(f"files[{index}] must be an object")
            action = item.get("action")
            required = {"path", "action"} if action == "delete" else {"path", "action", "content"}
            _require_exact_keys(item, required=required, label=f"files[{index}]")
            path = item["path"]
            content = item.get("content")
            if action not in {"create", "update", "delete"}:
                raise ContractError(f"files[{index}].action is invalid")
            if not isinstance(path, str):
                raise ContractError(f"files[{index}].path must be a string")
            path = normalise_relative_path(path)
            if action == "delete":
                content = None
            elif not isinstance(content, str):
                raise ContractError(f"files[{index}].content must be a string")
            operations.append(FileOperation(path=path, action=action, content=content))
        return PatchResponse(summary=summary.strip(), files=tuple(operations))

    if schema == ESCALATION_SCHEMA:
        _require_exact_keys(
            value,
            required={"schema_version", "reason_code", "message"},
            label="escalation response",
        )
        reason = value["reason_code"]
        message = value["message"]
        if reason not in ESCALATION_CODES:
            raise ContractError(f"invalid escalation reason_code: {reason!r}")
        if not isinstance(message, str) or not message.strip() or len(message) > 500:
            raise ContractError("escalation message must be 1-500 characters")
        return EscalationResponse(reason_code=reason, message=message.strip())

    raise ContractError(f"unsupported schema_version: {schema!r}")
