from __future__ import annotations

from pathlib import Path
from typing import Any

from nova_v12.inference.protocol import parse_agent_action
from nova_v12.patching import apply_operations, safe_resolve
from nova_v12.sandbox import SandboxRunner
from nova_v12.schemas import CommandSpec


class ToolEnvironment:
    """Deterministic tool environment used for multi-turn agent evaluation."""

    def __init__(self, root: Path, tests: list[CommandSpec], runner: SandboxRunner) -> None:
        self.root = root
        self.tests = tests
        self.runner = runner
        self.changed_files: set[str] = set()

    @property
    def tool_description(self) -> str:
        return (
            "Available tools: list_files, read_file, search_code, "
            "apply_operations, run_tests, finish. "
            'Return exactly one JSON object: {"tool": "name", "args": {...}}.'
        )

    def execute(self, raw_output: str) -> dict[str, Any]:
        try:
            action = parse_agent_action(raw_output)
        except Exception as exc:
            return {"ok": False, "error": f"invalid tool call: {exc}"}
        tool = action["tool"]
        args = action["args"]
        if tool == "list_files":
            relative = str(args.get("path", "."))
            directory = safe_resolve(self.root, relative)
            if not directory.is_dir():
                return {"ok": False, "error": "directory not found"}
            values = [
                str(path.relative_to(self.root))
                for path in sorted(directory.rglob("*"))
                if path.is_file()
            ]
            return {"ok": True, "files": values[:500]}
        if tool == "read_file":
            path = safe_resolve(self.root, str(args.get("path", "")))
            if not path.is_file():
                return {"ok": False, "error": "file not found"}
            return {"ok": True, "content": path.read_text(encoding="utf-8")}
        if tool == "search_code":
            query = str(args.get("query", ""))
            base = safe_resolve(self.root, str(args.get("path", ".")))
            matches: list[dict[str, Any]] = []
            for file in base.rglob("*"):
                if not file.is_file() or file.stat().st_size > 1_000_000:
                    continue
                try:
                    text = file.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                for index, line in enumerate(text.splitlines(), 1):
                    if query in line:
                        matches.append(
                            {"path": str(file.relative_to(self.root)), "line": index, "text": line}
                        )
                        if len(matches) >= 100:
                            return {"ok": True, "matches": matches}
            return {"ok": True, "matches": matches}
        if tool == "apply_operations":
            operations = args.get("operations")
            if not isinstance(operations, list):
                return {"ok": False, "error": "operations must be a list"}
            result = apply_operations(self.root, operations)
            self.changed_files.update(result.changed_files)
            return {"ok": result.ok, "changed_files": result.changed_files, "error": result.error}
        if tool == "run_tests":
            result = self.runner.verify(self.root, self.tests, sorted(self.changed_files))
            return result.to_dict()
        if tool == "finish":
            return {"ok": True, "finished": True, "summary": str(args.get("summary", ""))}
        return {"ok": False, "error": f"unknown tool: {tool}"}
