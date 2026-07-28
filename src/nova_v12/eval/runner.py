from __future__ import annotations

import json
from pathlib import Path

from nova_v12.eval.backends import GenerationBackend
from nova_v12.eval.tasks import load_tasks
from nova_v12.eval.tool_env import ToolEnvironment
from nova_v12.sandbox import SandboxRunner
from nova_v12.schemas import EvalTask, GenerationResult, write_jsonl


def format_prompt(task: EvalTask) -> str:
    if task.category == "fim":
        return (
            "Complete only the missing code between the prefix and suffix. Return code only.\n\n"
            f"PREFIX:\n{task.prefix}\n\nSUFFIX:\n{task.suffix}"
        )
    if task.category in {"debugging", "repository_editing"}:
        files = "\n\n".join(f"FILE: {item.path}\n{item.content}" for item in task.files)
        return (
            f"{task.prompt}\n\nRepository snapshot:\n{files}\n\n"
            "Return only JSON using this schema: "
            '{"status":"patch","operations":[{"action":"replace|create|write",'
            '"path":"relative/path","search":"exact old text","replace":"new text"}]}. '
            "Use minimal changes and only repository-relative paths."
        )
    if task.category == "instruction_following":
        return task.prompt
    return task.prompt


def run_tasks(
    backend: GenerationBackend,
    candidate_id: str,
    task_path: str | Path,
    output_path: str | Path,
    *,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    categories: set[str] | None = None,
) -> list[GenerationResult]:
    tasks = load_tasks(task_path)
    if categories:
        tasks = [task for task in tasks if task.category in categories]
    results: list[GenerationResult] = []
    sandbox = SandboxRunner()

    for task in tasks:
        try:
            if task.category == "tool_use":
                result = _run_tool_task(
                    backend,
                    candidate_id,
                    task,
                    sandbox,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                prompt = format_prompt(task)
                generate_for_task = getattr(backend, "generate_for_task", None)
                if callable(generate_for_task):
                    generated = generate_for_task(
                        task, prompt, max_tokens=max_tokens, temperature=temperature
                    )
                else:
                    generated = backend.generate(
                        prompt, max_tokens=max_tokens, temperature=temperature
                    )
                result = GenerationResult(
                    candidate_id=candidate_id,
                    task_id=task.id,
                    category=task.category,
                    raw_output=generated.text,
                    latency_seconds=generated.latency_seconds,
                    tokens_generated=generated.tokens_generated,
                    tokens_per_second=generated.tokens_per_second,
                    model_revision=generated.revision,
                    backend=backend.name,
                    task=task.to_dict(),
                )
        except Exception as exc:
            result = GenerationResult(
                candidate_id=candidate_id,
                task_id=task.id,
                category=task.category,
                raw_output="",
                error=f"{type(exc).__name__}: {exc}",
                backend=backend.name,
                task=task.to_dict(),
            )
        results.append(result)
        write_jsonl(output_path, (item.to_dict() for item in results))
    return results


def _run_tool_task(
    backend: GenerationBackend,
    candidate_id: str,
    task: EvalTask,
    sandbox: SandboxRunner,
    *,
    max_tokens: int,
    temperature: float,
) -> GenerationResult:
    temp, root = sandbox.make_workspace(task.files)
    interactions: list[dict] = []
    total_latency = 0.0
    total_tokens = 0
    outputs: list[str] = []
    environment = ToolEnvironment(root, task.tests, sandbox)
    transcript = f"{task.prompt}\n\n{environment.tool_description}"
    try:
        for _ in range(max(1, task.tool_budget or 8)):
            generated = backend.generate(transcript, max_tokens=max_tokens, temperature=temperature)
            total_latency += generated.latency_seconds
            total_tokens += generated.tokens_generated
            outputs.append(generated.text)
            observation = environment.execute(generated.text)
            interactions.append({"assistant": generated.text, "observation": observation})
            if observation.get("finished"):
                break
            transcript += (
                f"\n\nASSISTANT ACTION:\n{generated.text}\n\n"
                f"TOOL OBSERVATION:\n{json.dumps(observation, ensure_ascii=False)}\n\n"
                "Return the next single tool call JSON object."
            )
    finally:
        temp.cleanup()
    return GenerationResult(
        candidate_id=candidate_id,
        task_id=task.id,
        category=task.category,
        raw_output=outputs[-1] if outputs else "",
        latency_seconds=total_latency,
        tokens_generated=total_tokens,
        tokens_per_second=total_tokens / total_latency if total_latency else 0.0,
        backend=backend.name,
        task=task.to_dict(),
        interactions=interactions,
    )
