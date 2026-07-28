import json
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "legacy" / "v11" / "coding_agent"),
)

from nexus.nova_v12_backend import NovaV12Backend  # noqa: E402


class Backend:
    def __init__(self, output):
        self.output = output

    def generate(self, prompt):
        del prompt
        return self.output


def test_nexus_gets_proposal_only_after_v12_verification(tmp_path):
    (tmp_path / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "check.py").write_text(
        "from value import VALUE\nassert VALUE == 2\n",
        encoding="utf-8",
    )
    output = json.dumps(
        {
            "schema_version": "nova.patch.v1",
            "summary": "Correct the value",
            "files": [{"path": "value.py", "action": "update", "content": "VALUE = 2\n"}],
        }
    )
    adapter = NovaV12Backend(
        working_dir=str(tmp_path),
        backend=Backend(output),
    )
    result = adapter.run_task(
        {
            "task_id": "nexus-1",
            "instruction": "Set VALUE to 2.",
            "allowed_files": ["value.py"],
            "context_files": ["check.py"],
            "test_command": ["python", "check.py"],
        }
    )
    assert len(result.proposals) == 1
    assert result.proposals[0].args["_nova_guardrail"]["passed"]
    assert result.proposals[0].args["content"] == "VALUE = 2\n"
    assert (tmp_path / "value.py").read_text() == "VALUE = 1\n"


def test_nexus_gets_no_proposal_for_failed_patch(tmp_path):
    (tmp_path / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "check.py").write_text(
        "from value import VALUE\nassert VALUE == 2\n",
        encoding="utf-8",
    )
    output = json.dumps(
        {
            "schema_version": "nova.patch.v1",
            "summary": "Wrong value",
            "files": [{"path": "value.py", "action": "update", "content": "VALUE = 3\n"}],
        }
    )
    adapter = NovaV12Backend(
        working_dir=str(tmp_path),
        backend=Backend(output),
    )
    result = adapter.run_task(
        {
            "task_id": "nexus-2",
            "instruction": "Set VALUE to 2.",
            "allowed_files": ["value.py"],
            "context_files": ["check.py"],
            "test_command": ["python", "check.py"],
        }
    )
    assert not result.proposals
    assert "no mutations" in result.assistant_text
