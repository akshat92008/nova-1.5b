"""Execution-verified dataset builder for Nova V12."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .constants import (
    ALLOWED_LICENCES,
    DATASET_SCHEMA,
    MANIFEST_SCHEMA,
)
from .execution import run_test_command, verify_response
from .policy import normalise_relative_path, validate_task
from .protocol import parse_response
from .schema import AtomicTask, ContractError, EscalationResponse, PatchResponse

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)(?:api[_-]?key|secret|password)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
)

BENCHMARK_MARKERS = frozenset(
    {
        "apps",
        "bigcodebench",
        "humaneval",
        "humaneval+",
        "livecodebench",
        "mbpp",
        "mbpp+",
        "swe-bench",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ContractError(f"{path}:{line_number}: record must be an object")
            yield line_number, value


def contains_secret(value: Any) -> bool:
    text = canonical_json(value)
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def write_workspace(root: Path, files: dict[str, str]) -> None:
    for raw, content in files.items():
        path = normalise_relative_path(raw)
        if not isinstance(content, str):
            raise ContractError(f"workspace file content must be text: {path}")
        target = root.joinpath(*Path(path).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


@dataclass(frozen=True)
class Rejection:
    line: int
    task_id: str
    reason: str


def _normalise_source_record(
    value: dict[str, Any],
) -> tuple[AtomicTask, dict[str, str], PatchResponse, dict[str, Any], str, str]:
    if value.get("schema_version") not in (None, DATASET_SCHEMA):
        raise ContractError(f"unsupported dataset schema: {value.get('schema_version')!r}")
    task_value = value.get("task")
    if not isinstance(task_value, dict):
        raise ContractError("task must be an object")
    task = AtomicTask.from_dict(task_value)
    validate_task(task)

    files = value.get("files_before")
    if not isinstance(files, dict) or not all(
        isinstance(path, str) and isinstance(content, str) for path, content in files.items()
    ):
        raise ContractError("files_before must be a path-to-text object")
    for path in files:
        normalise_relative_path(path)

    raw_response = value.get("response")
    if isinstance(raw_response, dict):
        raw_response = canonical_json(raw_response)
    if not isinstance(raw_response, str):
        raise ContractError("response must be protocol JSON text or an object")
    parsed = parse_response(raw_response)
    if isinstance(parsed, EscalationResponse):
        raise ContractError("verified patch dataset cannot contain escalation responses")

    provenance = value.get("provenance")
    if not isinstance(provenance, dict):
        raise ContractError("provenance must be an object")
    required_provenance = {"source_repository", "source_commit", "generation_method"}
    missing = required_provenance - provenance.keys()
    if missing:
        raise ContractError(f"provenance missing keys: {sorted(missing)}")
    if not all(isinstance(provenance[key], str) and provenance[key] for key in required_provenance):
        raise ContractError("provenance fields must be non-empty strings")

    licence = str(value.get("licence", "")).lower()
    if licence not in ALLOWED_LICENCES:
        raise ContractError(f"licence is not allowlisted: {licence!r}")
    split = str(value.get("split", "train"))
    if split not in {"train", "validation", "held_out"}:
        raise ContractError(f"invalid split: {split!r}")

    benchmark = str(provenance.get("benchmark", "")).lower()
    if benchmark in BENCHMARK_MARKERS:
        raise ContractError(f"benchmark-contaminated source: {benchmark}")
    if contains_secret(value):
        raise ContractError("record appears to contain a secret")
    return task, files, parsed, provenance, licence, split


def verify_record(value: dict[str, Any], *, timeout_seconds: int) -> dict[str, Any]:
    """Validate and execute one source record, returning immutable evidence."""
    task, files, response, provenance, licence, split = _normalise_source_record(value)

    with tempfile.TemporaryDirectory(prefix="nova-v12-data-") as temp:
        workspace = Path(temp) / "workspace"
        workspace.mkdir()
        write_workspace(workspace, files)

        baseline = run_test_command(
            workspace,
            task.test_command,
            timeout_seconds=timeout_seconds,
        )
        changed, changed_lines, patched = verify_response(
            workspace,
            task,
            response,
            timeout_seconds=timeout_seconds,
        )

    if not patched.passed:
        raise ContractError(
            f"patched tests failed: exit={patched.exit_code} timeout={patched.timed_out}; "
            f"stderr={patched.stderr[-500:]}"
        )
    if task.task_kind == "repair" and baseline.passed:
        raise ContractError("repair example baseline already passes")
    if set(changed) != {operation.path for operation in response.files}:
        raise ContractError("verifier changed-file evidence is inconsistent")

    prompt_record = {
        "task": task.to_dict(),
        "files_before": files,
    }
    response_dict = response.to_dict()
    record_id = sha256_text(
        canonical_json(
            {
                "prompt": prompt_record,
                "response": response_dict,
                "source_repository": provenance["source_repository"],
                "source_commit": provenance["source_commit"],
            }
        )
    )
    return {
        "schema_version": DATASET_SCHEMA,
        "record_id": record_id,
        "split": split,
        "mode": "<|nova_patch|>",
        "prompt": prompt_record,
        "response": response_dict,
        "licence": licence,
        "provenance": provenance,
        "verification": {
            "verified": True,
            "baseline": baseline.to_dict(),
            "patched": patched.to_dict(),
            "changed_files": list(changed),
            "changed_lines": changed_lines,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def _validate_split_isolation(records: Iterable[dict[str, Any]]) -> None:
    repo_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        repo = record["provenance"]["source_repository"]
        repo_splits[repo].add(record["split"])
    leaked = {repo: splits for repo, splits in repo_splits.items() if len(splits) > 1}
    if leaked:
        preview = ", ".join(f"{repo}={sorted(splits)}" for repo, splits in list(leaked.items())[:5])
        raise ContractError(f"repository leakage across splits: {preview}")


def build_verified_dataset(
    source: Path,
    output_dir: Path,
    *,
    timeout_seconds: int = 60,
    fail_on_rejection: bool = True,
) -> dict[str, Any]:
    """Build verified JSONL shards plus a content-addressed manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    records: list[dict[str, Any]] = []
    rejections: list[Rejection] = []
    seen: set[str] = set()
    for line, value in iter_jsonl(source):
        task_id = str(value.get("task", {}).get("task_id", value.get("task", {}).get("id", "?")))
        try:
            record = verify_record(value, timeout_seconds=timeout_seconds)
            if record["record_id"] in seen:
                raise ContractError("duplicate verified record")
            seen.add(record["record_id"])
            records.append(record)
        except (ContractError, OSError) as exc:
            rejections.append(Rejection(line=line, task_id=task_id, reason=str(exc)))

    if not records:
        shutil.rmtree(staging)
        raise ContractError("no records passed execution verification")
    _validate_split_isolation(records)

    split_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    shard_paths: list[Path] = []
    for split in ("train", "validation", "held_out"):
        split_records = [record for record in records if record["split"] == split]
        if not split_records:
            continue
        path = staging / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in sorted(split_records, key=lambda item: item["record_id"]):
                handle.write(canonical_json(record) + "\n")
                split_counts[split] += 1
                mode_counts[record["mode"]] += 1
        shard_paths.append(path)

    rejection_path = staging / "rejections.jsonl"
    with rejection_path.open("w", encoding="utf-8") as handle:
        for rejection in rejections:
            handle.write(canonical_json(asdict(rejection)) + "\n")

    shards = [
        {
            "path": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "records": split_counts[path.stem],
        }
        for path in shard_paths
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": source.name,
            "sha256": sha256_file(source),
        },
        "verified_examples": len(records),
        "rejected_examples": len(rejections),
        "split_counts": dict(sorted(split_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "shards": shards,
        "verification_policy": {
            "max_changed_files": 3,
            "single_repair_attempt": True,
            "requires_passing_tests": True,
            "repair_requires_failing_baseline": True,
            "allowed_licences": sorted(ALLOWED_LICENCES),
            "benchmark_markers_blocked": sorted(BENCHMARK_MARKERS),
        },
    }
    manifest["manifest_sha256"] = sha256_text(canonical_json(manifest))
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if fail_on_rejection and rejections:
        rejection_copy = output_dir / "rejections.jsonl"
        shutil.copy2(rejection_path, rejection_copy)
        shutil.rmtree(staging)
        raise ContractError(
            f"{len(rejections)} record(s) rejected; inspect {rejection_copy} and rebuild"
        )

    for existing in output_dir.glob("*.jsonl"):
        existing.unlink()
    existing_manifest = output_dir / "manifest.json"
    if existing_manifest.exists():
        existing_manifest.unlink()
    for path in staging.iterdir():
        shutil.move(str(path), output_dir / path.name)
    staging.rmdir()
    return manifest


def validate_manifest(dataset_dir: Path, *, minimum_verified: int = 1) -> dict[str, Any]:
    """Verify manifest integrity before a training stage consumes data."""
    path = dataset_dir / "manifest.json"
    if not path.is_file():
        raise ContractError(f"missing dataset manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ContractError("unsupported dataset manifest schema")
    supplied_hash = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    expected_hash = sha256_text(canonical_json(unsigned))
    if supplied_hash != expected_hash:
        raise ContractError("dataset manifest hash is invalid")
    if int(manifest.get("verified_examples", 0)) < minimum_verified:
        raise ContractError(
            f"dataset has {manifest.get('verified_examples', 0)} verified examples; "
            f"minimum is {minimum_verified}"
        )
    if int(manifest.get("rejected_examples", 0)) != 0:
        raise ContractError("dataset manifest contains rejected examples")
    total = 0
    for shard in manifest.get("shards", []):
        shard_path = dataset_dir / shard["path"]
        if not shard_path.is_file():
            raise ContractError(f"missing dataset shard: {shard_path}")
        if sha256_file(shard_path) != shard["sha256"]:
            raise ContractError(f"dataset shard hash mismatch: {shard_path.name}")
        lines = sum(1 for _ in iter_jsonl(shard_path))
        if lines != shard["records"]:
            raise ContractError(f"dataset shard count mismatch: {shard_path.name}")
        total += lines
    if total != manifest["verified_examples"]:
        raise ContractError("dataset total does not match manifest")
    return manifest


def main() -> None:  # pragma: no cover - exercised by CLI smoke tests
    parser = argparse.ArgumentParser(description="Build Nova V12 verified training data")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--allow-rejections",
        action="store_true",
        help="Publish valid records while retaining rejection evidence",
    )
    args = parser.parse_args()
    try:
        manifest = build_verified_dataset(
            args.source,
            args.output_dir,
            timeout_seconds=args.timeout,
            fail_on_rejection=not args.allow_rejections,
        )
    except ContractError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
