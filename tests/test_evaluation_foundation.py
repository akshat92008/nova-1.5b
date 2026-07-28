import json

import pytest

from nova_v12.dataset import canonical_json, sha256_file, sha256_text
from nova_v12.evaluation import RecordedBackend, evaluate_backend, release_decision
from nova_v12.foundation import select_foundation
from nova_v12.schema import ContractError


def write_evaluation(directory, model_id, success, case_hash="cases", tasks=1_000):
    directory.mkdir()
    raw = directory / "raw_results.jsonl"
    raw.write_text('{"evidence":"present"}\n', encoding="utf-8")
    metrics = {
        "tasks": tasks,
        "task_success_rate": success,
        "first_attempt_success_rate": success - 0.05,
        "repair_success_rate": 0.05,
        "valid_protocol_rate": 0.99,
        "patch_application_rate": 0.97,
        "scope_compliance_rate": 1.0,
        "escalation_rate": 0.03,
        "median_changed_lines": 4,
        "p50_latency_seconds": 1.0,
        "p95_latency_seconds": 2.0,
        "raw_output_evidence_rate": 1.0,
    }
    value = {
        "schema_version": "nova.evaluation.v1",
        "created_at": "2026-07-28T00:00:00+00:00",
        "model": {"id": model_id, "revision": "rev"},
        "cases": {"path": "cases.jsonl", "sha256": case_hash},
        "raw_results": {"path": raw.name, "sha256": sha256_file(raw)},
        "metrics": metrics,
    }
    value["evaluation_sha256"] = sha256_text(canonical_json(value))
    path = directory / "evaluation.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_release_passes_measured_gates(tmp_path):
    baseline = write_evaluation(tmp_path / "base", "base", 0.76)
    candidate = write_evaluation(tmp_path / "candidate", "candidate", 0.80)
    result = release_decision(candidate, baseline)
    assert result["release_status"] == "passed"


def test_release_blocks_regression(tmp_path):
    baseline = write_evaluation(tmp_path / "base", "base", 0.80)
    candidate = write_evaluation(tmp_path / "candidate", "candidate", 0.76)
    result = release_decision(candidate, baseline)
    assert result["release_status"] == "blocked"
    assert not result["gates"]["no_success_regression_vs_base"]


def test_foundation_selects_execution_winner(tmp_path):
    first = write_evaluation(tmp_path / "one", "one", 0.78)
    second = write_evaluation(tmp_path / "two", "two", 0.82)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "candidates": {
                    "one": {"hf_id": "org/one", "revision": "1", "licence": "mit"},
                    "two": {"hf_id": "org/two", "revision": "2", "licence": "apache-2.0"},
                }
            }
        ),
        encoding="utf-8",
    )
    result = select_foundation([first, second], metadata)
    assert result["candidate_id"] == "two"


def test_foundation_requires_identical_cases(tmp_path):
    first = write_evaluation(tmp_path / "one", "one", 0.78, case_hash="a")
    second = write_evaluation(tmp_path / "two", "two", 0.82, case_hash="b")
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"candidates":{}}', encoding="utf-8")
    with pytest.raises(ContractError, match="identical"):
        select_foundation([first, second], metadata)


def test_evaluate_backend_preserves_raw_evidence(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {
                "task": {
                    "task_id": "eval-1",
                    "instruction": "Set VALUE to 2.",
                    "allowed_files": ["value.py"],
                    "context_files": ["check.py"],
                    "test_command": ["python", "check.py"],
                },
                "files_before": {
                    "value.py": "VALUE = 1\n",
                    "check.py": "from value import VALUE\nassert VALUE == 2\n",
                },
                "provenance": {
                    "source_repository": "private/eval-one",
                    "split": "held_out",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = json.dumps(
        {
            "schema_version": "nova.patch.v1",
            "summary": "Correct the value",
            "files": [{"path": "value.py", "action": "update", "content": "VALUE = 2\n"}],
        }
    )
    recorded = tmp_path / "outputs.jsonl"
    recorded.write_text(json.dumps({"raw_output": output}) + "\n", encoding="utf-8")
    report = evaluate_backend(
        RecordedBackend.from_jsonl(recorded),
        cases,
        tmp_path / "evaluation",
        model_id="recorded",
        model_revision="rev",
    )
    assert report["metrics"]["task_success_rate"] == 1.0
    assert report["metrics"]["scope_compliance_rate"] == 1.0
    assert (tmp_path / "evaluation" / "raw_results.jsonl").is_file()


def test_evaluation_rejects_non_held_out_case(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {
                "task": {
                    "task_id": "bad",
                    "instruction": "Do a thing.",
                    "allowed_files": ["x.py"],
                    "test_command": ["python", "test.py"],
                },
                "files_before": {},
                "provenance": {"source_repository": "repo", "split": "train"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    backend = RecordedBackend(iter(["{}"]))
    with pytest.raises(ContractError, match="not held_out"):
        evaluate_backend(
            backend,
            cases,
            tmp_path / "out",
            model_id="bad",
            model_revision="bad",
        )
