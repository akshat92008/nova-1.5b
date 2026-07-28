import json
from pathlib import Path

from nova_v12.runner import NovaRunner, OllamaBackend
from nova_v12.schema import AtomicTask


class SequenceBackend:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return next(self.outputs)


def make_workspace(root: Path):
    (root / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "check.py").write_text(
        "from value import VALUE\nassert VALUE == 2\n",
        encoding="utf-8",
    )


def task():
    return AtomicTask.from_dict(
        {
            "task_id": "runner-1",
            "instruction": "Set VALUE to 2.",
            "allowed_files": ["value.py"],
            "context_files": ["check.py"],
            "test_command": ["python", "check.py"],
        }
    )


def output(content, path="value.py"):
    return json.dumps(
        {
            "schema_version": "nova.patch.v1",
            "summary": "Set the requested value",
            "files": [{"path": path, "action": "update", "content": content}],
        }
    )


def test_pass_commits(tmp_path):
    make_workspace(tmp_path)
    backend = SequenceBackend(output("VALUE = 2\n"))
    result = NovaRunner(backend).run(task(), tmp_path)
    assert result.passed
    assert (tmp_path / "value.py").read_text() == "VALUE = 2\n"
    assert backend.calls == 1


def test_one_repair_then_commit(tmp_path):
    make_workspace(tmp_path)
    backend = SequenceBackend(output("VALUE = 3\n"), output("VALUE = 2\n"))
    result = NovaRunner(backend).run(task(), tmp_path)
    assert result.passed
    assert len(result.attempts) == 2
    assert (tmp_path / "value.py").read_text() == "VALUE = 2\n"


def test_two_failures_do_not_mutate(tmp_path):
    make_workspace(tmp_path)
    backend = SequenceBackend(output("VALUE = 3\n"), output("VALUE = 4\n"))
    result = NovaRunner(backend).run(task(), tmp_path)
    assert result.status == "failed"
    assert not result.committed
    assert (tmp_path / "value.py").read_text() == "VALUE = 1\n"


def test_scope_violation_does_not_mutate(tmp_path):
    make_workspace(tmp_path)
    backend = SequenceBackend(output("BAD = 1\n", "check.py"), output("BAD = 2\n", "check.py"))
    result = NovaRunner(backend).run(task(), tmp_path)
    assert result.status == "failed"
    assert result.attempts[0].protocol_valid
    assert not result.attempts[0].scope_valid
    assert (tmp_path / "check.py").read_text().startswith("from value")


def test_model_escalation_stops_without_retry(tmp_path):
    make_workspace(tmp_path)
    backend = SequenceBackend(
        json.dumps(
            {
                "schema_version": "nova.escalation.v1",
                "reason_code": "missing_context",
                "message": "Required API contract is not supplied.",
            }
        )
    )
    result = NovaRunner(backend).run(task(), tmp_path)
    assert result.status == "escalated"
    assert backend.calls == 1


def test_ollama_uses_exact_response_schema(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"response":"{}"}'

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("nova_v12.runner.urllib.request.urlopen", fake_urlopen)
    assert OllamaBackend("nova-test").generate("task") == "{}"
    assert "oneOf" in captured["payload"]["format"]
    assert captured["payload"]["options"]["temperature"] == 0
