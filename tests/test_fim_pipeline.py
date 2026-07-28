import json

import yaml

from nova_v12.data.fim import generate_fim_records
from nova_v12.data.pipeline import build_data


def test_fim_generation_round_trips():
    content = "def square(value):\n    result = value * value\n    return result\n"
    records = generate_fim_records(content, language="python", source_hash="a" * 64, count=1)
    assert records
    record = records[0]
    assert record.prefix + record.middle + record.suffix == content


def test_data_pipeline_filters_and_writes_manifest(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps(
            {
                "source": "test",
                "repository": "org/repo",
                "revision": "abc",
                "path": "src/math_utils.py",
                "licence": "MIT",
                "language": "python",
                "content": (
                    '"""Useful helpers."""\n\n'
                    "def add_values(left, right):\n"
                    "    return left + right\n"
                ),
            }
        )
        + "\n"
    )
    config = {
        "input": {"kind": "jsonl", "paths": [str(raw)]},
        "output": {
            "records": str(tmp_path / "out.jsonl"),
            "manifest": str(tmp_path / "manifest.json"),
            "dedup_db": str(tmp_path / "dedup.sqlite3"),
        },
        "filters": {"licence_allowlist": ["mit"], "reject_pii": True},
        "limits": {"max_records": 10},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    manifest = build_data(config_path)
    assert manifest["stats"]["accepted"] == 1
    assert (tmp_path / "out.jsonl").exists()
