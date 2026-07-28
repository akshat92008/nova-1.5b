import json

from nova_v12.data.contamination import ContaminationScanner
from nova_v12.data.dedup import SQLiteExactDeduplicator


def test_sqlite_dedup_is_resumable(tmp_path):
    path = tmp_path / "dedup.sqlite3"
    with SQLiteExactDeduplicator(path) as dedup:
        assert dedup.add("hello world")
        assert not dedup.add("hello   world")
    with SQLiteExactDeduplicator(path) as dedup:
        assert not dedup.add("hello world")


def test_contamination_scans_nested_fields(tmp_path):
    signatures = {"signatures": [{"benchmark": "bench", "id": "x", "text": "def secret_eval("}]}
    path = tmp_path / "signatures.json"
    path.write_text(json.dumps(signatures))
    scanner = ContaminationScanner.from_file(path)
    findings = scanner.scan(
        {"messages": [{"role": "user", "content": "Use def secret_eval(value):"}]}
    )
    assert findings
    assert findings[0].field.endswith(".content")
