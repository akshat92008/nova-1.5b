import json
from pathlib import Path

import pytest

from nova_v12.dataset import build_verified_dataset, validate_manifest
from nova_v12.schema import ContractError


def source_record(repo="example/repo", split="train"):
    return {
        "schema_version": "nova.dataset.v1",
        "task": {
            "task_id": f"data-{split}",
            "instruction": "Set VALUE to 2.",
            "allowed_files": ["value.py"],
            "context_files": ["check.py"],
            "test_command": ["python", "check.py"],
            "task_kind": "repair",
        },
        "files_before": {
            "value.py": "VALUE = 1\n",
            "check.py": "from value import VALUE\nassert VALUE == 2\n",
        },
        "response": {
            "schema_version": "nova.patch.v1",
            "summary": "Correct the value",
            "files": [{"path": "value.py", "action": "update", "content": "VALUE = 2\n"}],
        },
        "provenance": {
            "source_repository": repo,
            "source_commit": "abc123",
            "generation_method": "human",
        },
        "licence": "mit",
        "split": split,
    }


def write_jsonl(path: Path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_builds_and_revalidates_manifest(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "out"
    write_jsonl(source, [source_record()])
    manifest = build_verified_dataset(source, output)
    assert manifest["verified_examples"] == 1
    assert validate_manifest(output)["manifest_sha256"] == manifest["manifest_sha256"]


def test_rejects_unlicensed_record(tmp_path):
    source = tmp_path / "source.jsonl"
    record = source_record()
    record["licence"] = "unknown"
    write_jsonl(source, [record])
    with pytest.raises(ContractError, match="no records"):
        build_verified_dataset(source, tmp_path / "out")


def test_rejects_repair_with_passing_baseline(tmp_path):
    source = tmp_path / "source.jsonl"
    record = source_record()
    record["files_before"]["value.py"] = "VALUE = 2\n"
    write_jsonl(source, [record])
    with pytest.raises(ContractError, match="no records"):
        build_verified_dataset(source, tmp_path / "out")


def test_rejects_repository_leakage_across_splits(tmp_path):
    source = tmp_path / "source.jsonl"
    write_jsonl(
        source,
        [source_record(split="train"), source_record(split="held_out")],
    )
    with pytest.raises(ContractError, match="leakage"):
        build_verified_dataset(source, tmp_path / "out")
