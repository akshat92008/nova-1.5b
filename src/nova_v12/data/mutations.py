from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nova_v12.patching import apply_operations
from nova_v12.sandbox import SandboxRunner
from nova_v12.schemas import CommandSpec, FileSpec, load_jsonl, write_jsonl


@dataclass(slots=True)
class MutationVerification:
    id: str
    verified: bool
    reason: str
    record: dict[str, Any] | None
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_mutation_record(
    value: dict[str, Any], runner: SandboxRunner | None = None
) -> MutationVerification:
    runner = runner or SandboxRunner(timeout_seconds=60)
    record_id = str(value.get("id", "mutation"))
    try:
        files = [FileSpec.from_dict(item) for item in value.get("files", [])]
        tests = [CommandSpec.from_dict(item) for item in value.get("tests", [])]
        operation = value.get("mutation")
        if not files or not tests or not isinstance(operation, dict):
            raise ValueError("files, tests and mutation operation are required")
        if operation.get("action") != "replace":
            raise ValueError("only exact replace mutations are supported")
        search = str(operation.get("search", ""))
        replacement = str(operation.get("replace", ""))
        if not search or search == replacement:
            raise ValueError("mutation search and replacement must be non-empty and differ")

        temp, root = runner.make_workspace(files)
        try:
            baseline = runner.verify(root, tests, [])
            if not baseline.passed:
                return MutationVerification(
                    record_id,
                    False,
                    "baseline tests do not pass",
                    None,
                    {"baseline": baseline.to_dict()},
                )
            mutated = apply_operations(root, [operation])
            if not mutated.ok:
                return MutationVerification(
                    record_id,
                    False,
                    f"mutation did not apply: {mutated.error}",
                    None,
                    {"baseline": baseline.to_dict()},
                )
            mutation_result = runner.verify(root, tests, mutated.changed_files)
            if mutation_result.passed:
                return MutationVerification(
                    record_id,
                    False,
                    "mutation was not detected by tests",
                    None,
                    {"baseline": baseline.to_dict(), "mutated": mutation_result.to_dict()},
                )
            repair = {
                "action": "replace",
                "path": operation["path"],
                "search": replacement,
                "replace": search,
                "expected_count": int(operation.get("expected_count", 1)),
            }
            repaired = apply_operations(root, [repair])
            if not repaired.ok:
                return MutationVerification(
                    record_id,
                    False,
                    f"repair did not apply: {repaired.error}",
                    None,
                    {"baseline": baseline.to_dict(), "mutated": mutation_result.to_dict()},
                )
            restored = runner.verify(root, tests, repaired.changed_files)
            if not restored.passed:
                return MutationVerification(
                    record_id,
                    False,
                    "repair did not restore passing tests",
                    None,
                    {
                        "baseline": baseline.to_dict(),
                        "mutated": mutation_result.to_dict(),
                        "restored": restored.to_dict(),
                    },
                )
            buggy_files = []
            for file in files:
                # The workspace is restored now, so reconstruct the buggy content exactly.
                content = file.content
                if file.path == operation["path"]:
                    content = content.replace(
                        search, replacement, int(operation.get("expected_count", 1))
                    )
                buggy_files.append({"path": file.path, "content": content})
            task = str(
                value.get("task") or "Repair the failing repository and return a minimal patch."
            )
            training = {
                "id": record_id,
                "mode": "debug",
                "messages": [
                    {
                        "role": "user",
                        "content": task
                        + "\n\n"
                        + "\n\n".join(
                            f"FILE: {item['path']}\n{item['content']}" for item in buggy_files
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": __import__("json").dumps(
                            {"status": "patch", "operations": [repair]}, sort_keys=True
                        ),
                    },
                ],
                "provenance": dict(value.get("provenance", {})),
                "verification": {
                    "passed": True,
                    "baseline": baseline.to_dict(),
                    "mutated": mutation_result.to_dict(),
                    "restored": restored.to_dict(),
                },
                "repository_snapshot": str(value.get("repository_snapshot", "")),
            }
            return MutationVerification(
                record_id,
                True,
                "",
                training,
                {
                    "baseline": baseline.to_dict(),
                    "mutated": mutation_result.to_dict(),
                    "restored": restored.to_dict(),
                },
            )
        finally:
            temp.cleanup()
    except Exception as exc:
        return MutationVerification(record_id, False, f"{type(exc).__name__}: {exc}", None, {})


def verify_mutation_file(input_path: str | Path, output_path: str | Path) -> tuple[int, int]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for value in load_jsonl(input_path):
        result = verify_mutation_record(value)
        if result.verified and result.record:
            accepted.append(result.record)
        else:
            rejected.append(result.to_dict())
    write_jsonl(output_path, accepted)
    write_jsonl(str(output_path) + ".rejected.jsonl", rejected)
    return len(accepted), len(rejected)
