from __future__ import annotations

from dataclasses import dataclass
from typing import Any

VALID_MODES = {"code", "fim", "edit", "debug", "agent", "review", "explain", "refactor"}
EXECUTION_REQUIRED_MODES = {"edit", "debug", "agent"}


@dataclass(slots=True)
class ValidationReport:
    valid: bool
    errors: list[str]


def validate_sft_record(record: dict[str, Any]) -> ValidationReport:
    errors: list[str] = []
    if not str(record.get("id", "")).strip():
        errors.append("id is required")
    mode = str(record.get("mode", ""))
    if mode not in VALID_MODES:
        errors.append(f"unsupported mode: {mode}")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append("messages must contain at least user and assistant turns")
    else:
        roles = [item.get("role") for item in messages if isinstance(item, dict)]
        if "user" not in roles or not roles or roles[-1] != "assistant":
            errors.append("messages must contain a user turn and end with assistant")
        if any(
            not isinstance(item.get("content"), str) or not item.get("content", "").strip()
            for item in messages
            if isinstance(item, dict)
        ):
            errors.append("all messages require non-empty text content")
    if not isinstance(record.get("provenance"), dict):
        errors.append("provenance object is required")
    if mode in EXECUTION_REQUIRED_MODES:
        verification = record.get("verification")
        if not isinstance(verification, dict) or verification.get("passed") is not True:
            errors.append(f"mode {mode} requires passing execution verification")
    return ValidationReport(not errors, errors)


def validate_dpo_record(record: dict[str, Any]) -> ValidationReport:
    errors: list[str] = []
    for field in ("id", "prompt", "chosen", "rejected", "repository_snapshot"):
        if not str(record.get(field, "")).strip():
            errors.append(f"{field} is required")
    if record.get("chosen") == record.get("rejected"):
        errors.append("chosen and rejected must differ")
    chosen = record.get("chosen_evidence")
    rejected = record.get("rejected_evidence")
    if not isinstance(chosen, dict) or not isinstance(rejected, dict):
        errors.append("chosen_evidence and rejected_evidence are required")
    else:
        chosen_score = float(chosen.get("score", 1.0 if chosen.get("passed") else 0.0))
        rejected_score = float(rejected.get("score", 1.0 if rejected.get("passed") else 0.0))
        if chosen.get("passed") is not True:
            errors.append("chosen candidate must have passed verification")
        if chosen_score <= rejected_score:
            errors.append("chosen evidence must be stronger than rejected evidence")
    return ValidationReport(not errors, errors)
