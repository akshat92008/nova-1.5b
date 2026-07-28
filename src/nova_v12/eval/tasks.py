from __future__ import annotations

from pathlib import Path

from nova_v12.schemas import EvalTask, load_jsonl


def load_tasks(path: str | Path) -> list[EvalTask]:
    target = Path(path)
    files = sorted(target.glob("*.jsonl")) if target.is_dir() else [target]
    tasks: list[EvalTask] = []
    seen: set[str] = set()
    for file in files:
        for value in load_jsonl(file):
            task = EvalTask.from_dict(value)
            if task.id in seen:
                raise ValueError(f"duplicate task id: {task.id}")
            seen.add(task.id)
            tasks.append(task)
    return tasks


def validate_tasks(path: str | Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    try:
        tasks = load_tasks(path)
    except Exception as exc:
        return 0, [str(exc)]
    for task in tasks:
        if (
            task.category in {"code_generation", "debugging", "repository_editing", "fim"}
            and not task.tests
        ):
            errors.append(f"{task.id}: executable category has no tests")
        if task.category == "fim" and not (task.prefix or task.suffix):
            errors.append(f"{task.id}: FIM task requires prefix or suffix")
        if task.category in {"debugging", "repository_editing", "tool_use"} and not task.files:
            errors.append(f"{task.id}: repository task requires fixture files")
    return len(tasks), errors
