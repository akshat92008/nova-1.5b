"""Constrained Nova inference with one verified repair attempt."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from .constants import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_ATTEMPTS,
    MODE_TOKEN,
    REPAIR_TOKEN,
    RESPONSE_JSON_SCHEMA,
    SYSTEM_PROMPT,
)
from .execution import commit_verified_response, verify_response
from .policy import resolve_workspace_path, validate_response_scope, validate_task
from .protocol import parse_response
from .schema import (
    AtomicTask,
    AttemptEvidence,
    ContractError,
    EscalationResponse,
    PatchResponse,
    RunEvidence,
)


class GenerationBackend(Protocol):
    """Minimal backend contract used by NovaRunner."""

    def generate(self, prompt: str) -> str:
        """Return raw model text."""


class OllamaBackend:
    """Deterministic Ollama generation backend."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 300,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "format": RESPONSE_JSON_SCHEMA,
            "options": {
                "temperature": 0,
                "top_p": 1,
                "seed": 42,
                "num_predict": self.max_tokens,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama generation failed: {exc}") from exc
        output = value.get("response")
        if not isinstance(output, str):
            raise RuntimeError("Ollama response did not contain text")
        return output


def build_prompt(task: AtomicTask, workspace: Path) -> str:
    """Build a bounded, explicit task prompt from planner-owned context."""
    validate_task(task)
    files: list[dict[str, str | bool]] = []
    for raw in dict.fromkeys((*task.allowed_files, *task.context_files)):
        target = resolve_workspace_path(workspace, raw)
        if target.exists():
            if not target.is_file():
                raise ContractError(f"context target is not a file: {raw}")
            content = target.read_text(encoding="utf-8")
            files.append({"path": raw, "exists": True, "content": content})
        else:
            files.append({"path": raw, "exists": False, "content": ""})
    payload = {
        "task_id": task.task_id,
        "instruction": task.instruction,
        "allowed_files": list(task.allowed_files),
        "test_command": list(task.test_command),
        "repository_context": files,
    }
    return f"{MODE_TOKEN}\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"


def build_repair_prompt(
    original_prompt: str,
    prior_output: str,
    error: str,
) -> str:
    """Create the sole repair prompt with concrete verifier evidence."""
    repair = {
        "original_request": original_prompt,
        "previous_output": prior_output,
        "verification_error": error[-12_000:],
        "instruction": "Return one corrected protocol object. Do not expand scope.",
    }
    return f"{REPAIR_TOKEN}\n{json.dumps(repair, ensure_ascii=False, separators=(',', ':'))}"


class NovaRunner:
    """Generate, verify in a copy, retry once, then commit or escalate."""

    def __init__(
        self,
        backend: GenerationBackend,
        *,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.backend = backend
        self.timeout_seconds = timeout_seconds

    def run(self, task: AtomicTask, workspace: str | Path) -> RunEvidence:
        root = Path(workspace).resolve()
        validate_task(task)
        prompt = build_prompt(task, root)
        attempts: list[AttemptEvidence] = []
        final: PatchResponse | EscalationResponse | None = None

        for number in range(1, MAX_ATTEMPTS + 1):
            started = time.perf_counter()
            raw = ""
            protocol_valid = False
            scope_valid = False
            try:
                raw = self.backend.generate(prompt)
                parsed = parse_response(raw)
                protocol_valid = True
                if isinstance(parsed, EscalationResponse):
                    attempts.append(
                        AttemptEvidence(
                            attempt=number,
                            protocol_valid=True,
                            scope_valid=True,
                            patch_applied=False,
                            tests_passed=False,
                            raw_output=raw,
                            error=f"model escalated: {parsed.reason_code}",
                        )
                    )
                    final = parsed
                    return RunEvidence(
                        task_id=task.task_id,
                        status="escalated",
                        attempts=tuple(attempts),
                        committed=False,
                        final_response=final,
                    )

                validate_response_scope(task, parsed, root)
                scope_valid = True
                changed, changed_lines, command = verify_response(
                    root,
                    task,
                    parsed,
                    timeout_seconds=self.timeout_seconds,
                )
                if command.passed:
                    commit_verified_response(root, task, parsed)
                    attempts.append(
                        AttemptEvidence(
                            attempt=number,
                            protocol_valid=True,
                            scope_valid=True,
                            patch_applied=True,
                            tests_passed=True,
                            changed_files=changed,
                            changed_lines=changed_lines,
                            raw_output=raw,
                            command=command,
                        )
                    )
                    return RunEvidence(
                        task_id=task.task_id,
                        status="passed",
                        attempts=tuple(attempts),
                        committed=True,
                        final_response=parsed,
                        metadata={"elapsed_seconds": time.perf_counter() - started},
                    )

                error = (
                    f"tests failed (exit={command.exit_code}, timeout={command.timed_out})\n"
                    f"stdout:\n{command.stdout}\nstderr:\n{command.stderr}"
                )
                attempts.append(
                    AttemptEvidence(
                        attempt=number,
                        protocol_valid=True,
                        scope_valid=True,
                        patch_applied=True,
                        tests_passed=False,
                        changed_files=changed,
                        changed_lines=changed_lines,
                        raw_output=raw,
                        error=error,
                        command=command,
                    )
                )
            except ContractError as exc:
                error = str(exc)
                attempts.append(
                    AttemptEvidence(
                        attempt=number,
                        protocol_valid=protocol_valid,
                        scope_valid=scope_valid,
                        patch_applied=False,
                        tests_passed=False,
                        raw_output=raw,
                        error=error,
                    )
                )
            except Exception as exc:
                error = f"backend or verifier failure: {exc}"
                attempts.append(
                    AttemptEvidence(
                        attempt=number,
                        protocol_valid=False,
                        scope_valid=False,
                        patch_applied=False,
                        tests_passed=False,
                        raw_output=raw,
                        error=error,
                    )
                )

            if number < MAX_ATTEMPTS:
                prompt = build_repair_prompt(prompt, raw, error)

        final = EscalationResponse(
            reason_code="unsupported",
            message="Patch failed verification after the single allowed repair attempt.",
        )
        return RunEvidence(
            task_id=task.task_id,
            status="failed",
            attempts=tuple(attempts),
            committed=False,
            final_response=final,
        )
