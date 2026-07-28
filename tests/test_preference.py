import json

import pytest

from nova_v12.preference import (
    build_preference_dataset,
    validate_preference_manifest,
)
from nova_v12.schema import ContractError


def pair():
    return {
        "task": {
            "task_id": "pair-1",
            "instruction": "Set VALUE to 2.",
            "allowed_files": ["value.py"],
            "context_files": ["check.py"],
            "test_command": ["python", "check.py"],
        },
        "files_before": {
            "value.py": "VALUE = 1\n",
            "check.py": "from value import VALUE\nassert VALUE == 2\n",
        },
        "chosen": {
            "schema_version": "nova.patch.v1",
            "summary": "Use the correct value",
            "files": [{"path": "value.py", "action": "update", "content": "VALUE = 2\n"}],
        },
        "rejected": {
            "schema_version": "nova.patch.v1",
            "summary": "Use the wrong value",
            "files": [{"path": "value.py", "action": "update", "content": "VALUE = 3\n"}],
        },
        "licence": "mit",
        "provenance": {
            "source_repository": "example/repo",
            "source_commit": "abc123",
            "generation_method": "model_candidates",
        },
    }


def test_builds_execution_ranked_pair(tmp_path):
    source = tmp_path / "pairs.jsonl"
    source.write_text(json.dumps(pair()) + "\n", encoding="utf-8")
    output = tmp_path / "output"
    manifest = build_preference_dataset(source, output)
    assert manifest["pairs"] == 1
    assert validate_preference_manifest(output)["manifest_sha256"] == manifest["manifest_sha256"]


def test_rejects_when_both_candidates_pass(tmp_path):
    source = tmp_path / "pairs.jsonl"
    value = pair()
    value["rejected"]["files"][0]["content"] = "VALUE = 2\n"
    source.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="also passes"):
        build_preference_dataset(source, tmp_path / "output")
