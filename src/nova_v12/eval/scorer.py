from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from nova_v12.inference.protocol import parse_patch_response
from nova_v12.patching import apply_operations, apply_unified_diff, extract_code
from nova_v12.sandbox import SandboxRunner
from nova_v12.schemas import CheckResult, EvalTask, GenerationResult, TaskScore, load_jsonl


def _weighted(checks: list[CheckResult]) -> float:
    total = sum(item.weight for item in checks)
    return sum(item.weight for item in checks if item.passed) / total if total else 0.0


def _output_path(task: EvalTask) -> str:
    return str(
        task.metadata.get("output_path")
        or {
            "python": "solution.py",
            "javascript": "solution.js",
            "typescript": "solution.ts",
            "go": "solution.go",
            "rust": "src/lib.rs",
            "java": "Solution.java",
            "cpp": "solution.cpp",
            "c++": "solution.cpp",
        }.get(task.language, "solution.txt")
    )


def score_result(result: GenerationResult, runner: SandboxRunner | None = None) -> TaskScore:
    runner = runner or SandboxRunner()
    if result.error:
        return TaskScore(
            result.candidate_id, result.task_id, result.category, 0.0, [], error=result.error
        )
    if not result.task:
        return TaskScore(
            result.candidate_id,
            result.task_id,
            result.category,
            0.0,
            [],
            error="task payload missing",
        )
    task = EvalTask.from_dict(result.task)
    try:
        if task.category == "instruction_following":
            return _score_instruction(result, task, runner)
        if task.category == "tool_use":
            return _score_tool_use(result, task, runner)
        return _score_executable(result, task, runner)
    except Exception as exc:
        return TaskScore(
            result.candidate_id,
            result.task_id,
            result.category,
            0.0,
            [],
            error=f"{type(exc).__name__}: {exc}",
        )


def _score_executable(result: GenerationResult, task: EvalTask, runner: SandboxRunner) -> TaskScore:
    temp, root = runner.make_workspace(task.files)
    checks: list[CheckResult] = []
    changed: list[str] = []
    try:
        if task.category in {"code_generation", "fim"}:
            code = extract_code(result.raw_output, task.language)
            if task.category == "fim":
                code = task.prefix + code + task.suffix
            target = root / _output_path(task)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(code, encoding="utf-8")
            changed = [str(target.relative_to(root))]
            checks.append(CheckResult("non_empty_output", bool(code.strip()), 0.1))
        else:
            try:
                manifest = parse_patch_response(result.raw_output)
                if manifest.get("status") != "patch":
                    checks.append(
                        CheckResult("patch_status", False, 0.2, str(manifest.get("status")))
                    )
                    return TaskScore(
                        result.candidate_id, task.id, task.category, _weighted(checks), checks
                    )
                applied = apply_operations(root, manifest["operations"])
            except Exception:
                applied = apply_unified_diff(root, extract_code(result.raw_output))
            checks.append(CheckResult("patch_applies", applied.ok, 0.25, applied.error))
            if not applied.ok:
                return TaskScore(
                    result.candidate_id, task.id, task.category, _weighted(checks), checks
                )
            changed = applied.changed_files
            if task.expected_files_modified:
                allowed = set(task.expected_files_modified)
                actual = set(changed)
                checks.append(
                    CheckResult(
                        "authorised_files_only",
                        actual.issubset(allowed),
                        0.15,
                        f"expected subset of {sorted(allowed)}, got {sorted(actual)}",
                    )
                )
                checks.append(
                    CheckResult(
                        "required_files_touched",
                        bool(actual & allowed),
                        0.05,
                        f"changed {sorted(actual)}",
                    )
                )

        verification = runner.verify(root, task.tests, changed)
        checks.append(CheckResult("tests_pass", verification.passed, 0.65, verification.error))
        return TaskScore(
            result.candidate_id,
            task.id,
            task.category,
            _weighted(checks),
            checks,
            execution=verification.to_dict(),
        )
    finally:
        temp.cleanup()


def _score_instruction(
    result: GenerationResult, task: EvalTask, runner: SandboxRunner
) -> TaskScore:
    checks: list[CheckResult] = []
    text = result.raw_output.strip()
    for constraint in task.constraints:
        passed = _evaluate_constraint(text, constraint.kind, constraint.value)
        if constraint.negate:
            passed = not passed
        checks.append(
            CheckResult(f"constraint:{constraint.kind}", passed, 1.0, repr(constraint.value))
        )
    execution: dict[str, Any] = {}
    if task.tests:
        temp, root = runner.make_workspace(task.files)
        try:
            code = extract_code(text, task.language)
            target = root / _output_path(task)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(code, encoding="utf-8")
            verified = runner.verify(root, task.tests, [str(target.relative_to(root))])
            checks.append(CheckResult("tests_pass", verified.passed, 2.0, verified.error))
            execution = verified.to_dict()
        finally:
            temp.cleanup()
    return TaskScore(
        result.candidate_id, task.id, task.category, _weighted(checks), checks, execution
    )


def _evaluate_constraint(text: str, kind: str, value: Any) -> bool:
    if kind == "valid_json":
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False
    if kind == "exact_json_fields":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return False
        return isinstance(payload, dict) and set(payload) == set(value)
    if kind == "no_markdown":
        return "```" not in text and not re.search(r"^#{1,6}\s", text, re.MULTILINE)
    if kind == "contains":
        return str(value) in text
    if kind == "not_contains":
        return str(value) not in text
    if kind == "regex":
        return re.search(str(value), text, re.MULTILINE | re.DOTALL) is not None
    if kind == "one_line":
        return len([line for line in text.splitlines() if line.strip()]) == 1
    if kind == "python_ast":
        try:
            ast.parse(extract_code(text, "python"))
            return True
        except SyntaxError:
            return False
    raise ValueError(f"unknown constraint kind: {kind}")


def _score_tool_use(result: GenerationResult, task: EvalTask, runner: SandboxRunner) -> TaskScore:
    temp, root = runner.make_workspace(task.files)
    checks: list[CheckResult] = []
    changed: list[str] = []
    tools: list[str] = []
    try:
        for turn in result.interactions:
            raw = str(turn.get("assistant", ""))
            try:
                from nova_v12.inference.protocol import parse_agent_action

                action = parse_agent_action(raw)
            except Exception:
                checks.append(CheckResult("valid_tool_call", False, 0.1))
                continue
            tools.append(action["tool"])
            checks.append(CheckResult("valid_tool_call", True, 0.1))
            if action["tool"] == "apply_operations":
                applied = apply_operations(root, action["args"].get("operations", []))
                if applied.ok:
                    changed.extend(applied.changed_files)
                checks.append(CheckResult("patch_applies", applied.ok, 0.2, applied.error))
        verification = runner.verify(root, task.tests, sorted(set(changed)))
        checks.append(CheckResult("tests_pass", verification.passed, 0.6, verification.error))
        if task.metadata.get("required_tools"):
            required = set(task.metadata["required_tools"])
            checks.append(
                CheckResult("required_tools_used", required.issubset(set(tools)), 0.1, str(tools))
            )
        return TaskScore(
            result.candidate_id,
            task.id,
            task.category,
            _weighted(checks),
            checks,
            execution=verification.to_dict(),
        )
    finally:
        temp.cleanup()


def score_results(input_path: str | Path) -> tuple[list[TaskScore], dict[str, Any]]:
    scores = [score_result(GenerationResult.from_dict(value)) for value in load_jsonl(input_path)]
    by_candidate: dict[str, list[TaskScore]] = defaultdict(list)
    for item in scores:
        by_candidate[item.candidate_id].append(item)
    summary: dict[str, Any] = {}
    for candidate, values in by_candidate.items():
        categories: dict[str, list[float]] = defaultdict(list)
        for value in values:
            categories[value.category].append(value.score)
        summary[candidate] = {
            "overall": sum(value.score for value in values) / len(values) if values else 0.0,
            "tasks": len(values),
            "errors": sum(1 for value in values if value.error),
            "categories": {
                name: sum(category_values) / len(category_values)
                for name, category_values in sorted(categories.items())
            },
        }
    return scores, summary
