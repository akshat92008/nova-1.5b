import json

import pytest
import yaml

from nova_v12.dataset import (
    build_verified_dataset,
    canonical_json,
    sha256_text,
)
from nova_v12.preference import build_preference_dataset
from nova_v12.release_training import create_run_manifest, load_foundation_lock
from nova_v12.schema import ContractError


def patch_record(task_id, repo, split):
    return {
        "task": {
            "task_id": task_id,
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


def preference_pair():
    value = patch_record("pair", "repo/pair", "train")
    value["chosen"] = value.pop("response")
    value["rejected"] = {
        "schema_version": "nova.patch.v1",
        "summary": "Wrong value",
        "files": [{"path": "value.py", "action": "update", "content": "VALUE = 3\n"}],
    }
    value.pop("split")
    return value


def write_lines(path, values):
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def foundation_lock(path):
    value = {
        "schema_version": "nova.foundation-lock.v1",
        "decision_status": "locked",
        "candidate_id": "candidate",
        "hf_id": "org/model",
        "revision": "immutable",
        "licence": "mit",
        "evaluation_sha256": "evidence",
    }
    value["lock_sha256"] = sha256_text(canonical_json(value))
    path.write_text(json.dumps(value), encoding="utf-8")


def test_training_plan_requires_verified_inputs(tmp_path):
    source = tmp_path / "sft-source.jsonl"
    write_lines(
        source,
        [
            patch_record("train", "repo/train", "train"),
            patch_record("validation", "repo/validation", "validation"),
        ],
    )
    dataset = tmp_path / "sft"
    build_verified_dataset(source, dataset)

    pair_source = tmp_path / "pair-source.jsonl"
    write_lines(pair_source, [preference_pair()])
    preferences = tmp_path / "preferences"
    build_preference_dataset(pair_source, preferences)

    lock = tmp_path / "foundation.lock.json"
    foundation_lock(lock)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "nova.training-config.v1",
                "foundation": {"trust_remote_code": False},
                "data": {
                    "minimum_verified_examples": 2,
                    "minimum_preference_pairs": 1,
                    "preference_dir": str(preferences),
                },
            }
        ),
        encoding="utf-8",
    )
    result = create_run_manifest(
        config,
        lock,
        dataset,
        tmp_path / "run",
    )
    assert result["status"] == "planned"
    assert [stage["name"] for stage in result["stages"]] == ["stage2_sft", "stage4_dpo"]


def test_foundation_lock_rejects_tampering(tmp_path):
    path = tmp_path / "foundation.lock.json"
    foundation_lock(path)
    value = json.loads(path.read_text())
    value["hf_id"] = "attacker/replacement"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ContractError, match="hash"):
        load_foundation_lock(path)
