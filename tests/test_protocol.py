import json

import pytest

from nova_v12.constants import ESCALATION_SCHEMA, PATCH_SCHEMA
from nova_v12.protocol import parse_response
from nova_v12.schema import ContractError, EscalationResponse, PatchResponse


def patch(**overrides):
    value = {
        "schema_version": PATCH_SCHEMA,
        "summary": "Fix the boundary check",
        "files": [{"path": "src/math.py", "action": "update", "content": "x = 1\n"}],
    }
    value.update(overrides)
    return json.dumps(value)


def test_parses_patch():
    result = parse_response(patch())
    assert isinstance(result, PatchResponse)
    assert result.files[0].path == "src/math.py"


def test_parses_escalation():
    result = parse_response(
        json.dumps(
            {
                "schema_version": ESCALATION_SCHEMA,
                "reason_code": "missing_context",
                "message": "The target file was not provided.",
            }
        )
    )
    assert isinstance(result, EscalationResponse)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        "```json\n{}\n```",
        '{"schema_version":"nova.patch.v1"} trailing',
        "[]",
    ],
)
def test_rejects_non_protocol(raw):
    with pytest.raises(ContractError):
        parse_response(raw)


def test_rejects_unknown_top_level_key():
    with pytest.raises(ContractError, match="unknown keys"):
        parse_response(patch(confidence=0.99))


def test_rejects_unknown_file_key():
    value = json.loads(patch())
    value["files"][0]["diff"] = "no"
    with pytest.raises(ContractError, match="unknown keys"):
        parse_response(json.dumps(value))


@pytest.mark.parametrize("path", ["/tmp/x", "../x", "src/../x", ".git/config", r"src\\x.py"])
def test_rejects_unsafe_paths(path):
    value = json.loads(patch())
    value["files"][0]["path"] = path
    with pytest.raises(ContractError):
        parse_response(json.dumps(value))


def test_delete_must_omit_content():
    value = json.loads(patch())
    value["files"][0] = {"path": "src/math.py", "action": "delete", "content": ""}
    with pytest.raises(ContractError, match="unknown keys"):
        parse_response(json.dumps(value))
