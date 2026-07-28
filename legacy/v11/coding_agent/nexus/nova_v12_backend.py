"""Nexus adapter for the Nova V12 atomic patch contract.

Unlike the legacy Nova backend, this adapter never asks the model to infer scope
or tests from prose. The Ceiling planner must provide an AtomicTask-compatible
object. NovaRunner validates the patch in a temporary repository copy; Nexus
receives guarded tool proposals only after the tests pass.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from nexus.nova_backend import (
    NovaBackendError,
    NovaBackendResult,
    NovaToolProposal,
)
from nova_v12.execution import copy_workspace
from nova_v12.runner import GenerationBackend, NovaRunner, OllamaBackend
from nova_v12.schema import AtomicTask, PatchResponse


class NovaV12Backend:
    """Convert a verified V12 result into Nexus safety-tagged file proposals."""

    def __init__(
        self,
        model: str = "nova-v12:q4_k_m",
        working_dir: str | None = None,
        *,
        backend: GenerationBackend | None = None,
        ollama_url: str = "http://localhost:11434",
        timeout_seconds: int = 60,
    ):
        self.model = model
        self.working_dir = Path(working_dir or ".").resolve()
        self.backend = backend or OllamaBackend(model, base_url=ollama_url)
        self.timeout_seconds = timeout_seconds

    def run_task(self, task_value: AtomicTask | dict[str, Any]) -> NovaBackendResult:
        task = (
            task_value if isinstance(task_value, AtomicTask) else AtomicTask.from_dict(task_value)
        )
        with tempfile.TemporaryDirectory(prefix="nexus-nova-v12-") as temp:
            verification = Path(temp) / "workspace"
            copy_workspace(self.working_dir, verification)
            evidence = NovaRunner(
                self.backend,
                timeout_seconds=self.timeout_seconds,
            ).run(task, verification)

        raw_output = "\n\n".join(
            f"[ATTEMPT {attempt.attempt}]\n{attempt.raw_output}" for attempt in evidence.attempts
        )
        guardrail = json.dumps(evidence.to_dict(), ensure_ascii=False, sort_keys=True)
        if not evidence.passed or not isinstance(evidence.final_response, PatchResponse):
            return NovaBackendResult(
                raw_output=raw_output,
                assistant_text=(
                    "Nova V12 escalated or failed verification; Nexus received no mutations."
                ),
                guardrail_output=guardrail,
                proposals=[],
            )

        proposals: list[NovaToolProposal] = []
        for operation in evidence.final_response.files:
            if operation.action == "delete":
                raise NovaBackendError(
                    "Nova V12 produced a verified delete, but Nexus does not expose a "
                    "guarded delete-file proposal yet."
                )
            assert operation.content is not None
            summary = (
                f"Nova V12 verified task {task.task_id}; tests passed; "
                f"attempts={len(evidence.attempts)}; file={operation.path}"
            )
            proposals.append(
                NovaToolProposal(
                    name="write_file",
                    args={
                        "path": operation.path,
                        "content": operation.content,
                        "_nova_guardrail": {
                            "passed": True,
                            "schema_version": evidence.schema_version,
                            "task_id": task.task_id,
                            "summary": summary,
                            "evidence": evidence.to_dict(),
                        },
                    },
                    source_path=operation.path,
                    guardrail_summary=summary,
                )
            )
        return NovaBackendResult(
            raw_output=raw_output,
            assistant_text=(f"Nova V12 verified {len(proposals)} Nexus file proposal(s)."),
            guardrail_output=guardrail,
            proposals=proposals,
        )
