"""Build execution-ranked DPO pairs from the same atomic task contract."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import ALLOWED_LICENCES, SYSTEM_PROMPT
from .dataset import (
    canonical_json,
    contains_secret,
    iter_jsonl,
    sha256_file,
    sha256_text,
    write_workspace,
)
from .execution import verify_response
from .protocol import parse_response
from .schema import AtomicTask, ContractError, EscalationResponse, PatchResponse

PREFERENCE_SCHEMA = "nova.preference.v1"
PREFERENCE_MANIFEST_SCHEMA = "nova.preference-manifest.v1"


def _parse_patch(value: Any, label: str) -> PatchResponse:
    raw = canonical_json(value) if isinstance(value, dict) else value
    if not isinstance(raw, str):
        raise ContractError(f"{label} must be protocol JSON")
    parsed = parse_response(raw)
    if isinstance(parsed, EscalationResponse):
        raise ContractError(f"{label} cannot be an escalation")
    return parsed


def verify_pair(value: dict[str, Any], timeout_seconds: int = 60) -> dict[str, Any]:
    task_value = value.get("task")
    files = value.get("files_before")
    provenance = value.get("provenance")
    licence = str(value.get("licence", "")).lower()
    if not isinstance(task_value, dict):
        raise ContractError("task must be an object")
    task = AtomicTask.from_dict(task_value)
    if not isinstance(files, dict) or not all(
        isinstance(path, str) and isinstance(content, str) for path, content in files.items()
    ):
        raise ContractError("files_before must be a text map")
    if not isinstance(provenance, dict):
        raise ContractError("provenance must be an object")
    if licence not in ALLOWED_LICENCES:
        raise ContractError(f"licence is not allowlisted: {licence!r}")
    if contains_secret(value):
        raise ContractError("pair appears to contain a secret")
    chosen = _parse_patch(value.get("chosen"), "chosen")
    rejected = _parse_patch(value.get("rejected"), "rejected")
    if chosen == rejected:
        raise ContractError("chosen and rejected responses are identical")

    with tempfile.TemporaryDirectory(prefix="nova-v12-dpo-") as temp:
        workspace = Path(temp) / "workspace"
        workspace.mkdir()
        write_workspace(workspace, files)
        chosen_files, chosen_lines, chosen_command = verify_response(
            workspace,
            task,
            chosen,
            timeout_seconds=timeout_seconds,
        )
        try:
            rejected_files, rejected_lines, rejected_command = verify_response(
                workspace,
                task,
                rejected,
                timeout_seconds=timeout_seconds,
            )
            rejected_applied = True
        except ContractError as exc:
            rejected_files, rejected_lines, rejected_command = (), 0, None
            rejected_applied = False
            rejected_error = str(exc)

    if not chosen_command.passed:
        raise ContractError("chosen response does not pass tests")
    if rejected_applied and rejected_command and rejected_command.passed:
        raise ContractError("rejected response also passes tests")
    prompt_payload = {"task": task.to_dict(), "files_before": files}
    result = {
        "schema_version": PREFERENCE_SCHEMA,
        "pair_id": sha256_text(
            canonical_json(
                {
                    "prompt": prompt_payload,
                    "chosen": chosen.to_dict(),
                    "rejected": rejected.to_dict(),
                }
            )
        ),
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"<|nova_patch|>\n{canonical_json(prompt_payload)}",
            },
        ],
        "chosen": [{"role": "assistant", "content": canonical_json(chosen.to_dict())}],
        "rejected": [{"role": "assistant", "content": canonical_json(rejected.to_dict())}],
        "ranking_evidence": {
            "chosen": {
                "changed_files": list(chosen_files),
                "changed_lines": chosen_lines,
                "command": chosen_command.to_dict(),
            },
            "rejected": {
                "patch_applied": rejected_applied,
                "changed_files": list(rejected_files),
                "changed_lines": rejected_lines,
                "command": rejected_command.to_dict() if rejected_command else None,
                "error": rejected_error if not rejected_applied else None,
            },
        },
        "licence": licence,
        "provenance": provenance,
    }
    return result


def build_preference_dataset(
    source: Path,
    output_dir: Path,
    *,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "preferences.jsonl"
    staging = output_dir / ".preferences.staging.jsonl"
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line, value in iter_jsonl(source):
        try:
            record = verify_pair(value, timeout_seconds)
        except ContractError as exc:
            raise ContractError(f"{source}:{line}: {exc}") from exc
        if record["pair_id"] in seen:
            raise ContractError(f"{source}:{line}: duplicate preference pair")
        seen.add(record["pair_id"])
        records.append(record)
    if not records:
        raise ContractError("no verified preference pairs")
    with staging.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item["pair_id"]):
            handle.write(canonical_json(record) + "\n")
    shutil.move(staging, output)
    manifest = {
        "schema_version": PREFERENCE_MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256_file(source),
        "pairs": len(records),
        "shard": {
            "path": output.name,
            "sha256": sha256_file(output),
        },
        "policy": {
            "chosen_must_pass": True,
            "rejected_must_fail": True,
            "execution_ranked": True,
        },
    }
    manifest["manifest_sha256"] = sha256_text(canonical_json(manifest))
    (output_dir / "preference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_preference_manifest(
    directory: Path,
    *,
    minimum_pairs: int = 1,
) -> dict[str, Any]:
    path = directory / "preference_manifest.json"
    if not path.is_file():
        raise ContractError(f"missing preference manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != PREFERENCE_MANIFEST_SCHEMA:
        raise ContractError("unsupported preference manifest")
    supplied = value.get("manifest_sha256")
    unsigned = dict(value)
    unsigned.pop("manifest_sha256", None)
    if supplied != sha256_text(canonical_json(unsigned)):
        raise ContractError("preference manifest hash is invalid")
    if int(value.get("pairs", 0)) < minimum_pairs:
        raise ContractError(
            f"preference data has {value.get('pairs', 0)} pairs; minimum is {minimum_pairs}"
        )
    shard = directory / value["shard"]["path"]
    if sha256_file(shard) != value["shard"]["sha256"]:
        raise ContractError("preference shard hash mismatch")
    if sum(1 for _ in iter_jsonl(shard)) != value["pairs"]:
        raise ContractError("preference pair count mismatch")
    return value


def main() -> None:  # pragma: no cover - exercised by CLI smoke tests
    parser = argparse.ArgumentParser(description="Build Nova V12 execution-ranked DPO data")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    try:
        result = build_preference_dataset(
            args.source,
            args.output_dir,
            timeout_seconds=args.timeout,
        )
    except (ContractError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
