"""Held-out atomic-task evaluation and release gates for Nova V12."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .constants import RELEASE_SCHEMA
from .dataset import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    sha256_text,
    write_workspace,
)
from .runner import GenerationBackend, NovaRunner, OllamaBackend
from .schema import AtomicTask, ContractError


@dataclass
class RecordedBackend:
    """Deterministic backend for auditing previously captured raw outputs."""

    outputs: Iterator[str]

    @classmethod
    def from_jsonl(cls, path: Path) -> "RecordedBackend":
        values: list[str] = []
        for _line, value in iter_jsonl(path):
            raw = value.get("raw_output", value.get("response"))
            if isinstance(raw, dict):
                raw = canonical_json(raw)
            if not isinstance(raw, str):
                raise ContractError("recorded output must contain raw_output text")
            values.append(raw)
        return cls(iter(values))

    def generate(self, prompt: str) -> str:
        del prompt
        try:
            return next(self.outputs)
        except StopIteration as exc:
            raise RuntimeError("recorded backend ran out of outputs") from exc


def load_eval_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    repositories: dict[str, set[str]] = {}
    for line, value in iter_jsonl(path):
        task_value = value.get("task")
        files = value.get("files_before")
        provenance = value.get("provenance")
        if not isinstance(task_value, dict):
            raise ContractError(f"{path}:{line}: task must be an object")
        task = AtomicTask.from_dict(task_value)
        if task.task_id in seen:
            raise ContractError(f"{path}:{line}: duplicate task_id {task.task_id}")
        seen.add(task.task_id)
        if not isinstance(files, dict) or not all(
            isinstance(key, str) and isinstance(content, str) for key, content in files.items()
        ):
            raise ContractError(f"{path}:{line}: files_before must be a text map")
        if not isinstance(provenance, dict):
            raise ContractError(f"{path}:{line}: provenance must be an object")
        repo = provenance.get("source_repository")
        split = provenance.get("split", "held_out")
        if not isinstance(repo, str) or not repo:
            raise ContractError(f"{path}:{line}: source_repository is required")
        repositories.setdefault(repo, set()).add(str(split))
        if split != "held_out":
            raise ContractError(f"{path}:{line}: evaluation case is not held_out")
        cases.append({"task": task, "files_before": files, "provenance": provenance})
    if not cases:
        raise ContractError("evaluation set is empty")
    return cases


def _percent(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for item in results if item["evidence"]["status"] == "passed")
    escalated = sum(1 for item in results if item["evidence"]["status"] == "escalated")
    protocol_valid = sum(
        1
        for item in results
        if item["evidence"]["attempts"] and item["evidence"]["attempts"][0]["protocol_valid"]
    )
    patch_applied = sum(
        1
        for item in results
        if any(attempt["patch_applied"] for attempt in item["evidence"]["attempts"])
    )
    scope_compliant = sum(
        1
        for item in results
        if all(attempt["scope_valid"] for attempt in item["evidence"]["attempts"])
    )
    repaired = sum(
        1
        for item in results
        if item["evidence"]["status"] == "passed" and len(item["evidence"]["attempts"]) == 2
    )
    first_pass = sum(
        1
        for item in results
        if item["evidence"]["status"] == "passed" and len(item["evidence"]["attempts"]) == 1
    )
    changed_lines = [
        attempt["changed_lines"]
        for item in results
        for attempt in item["evidence"]["attempts"]
        if attempt["tests_passed"]
    ]
    latencies = [float(item["latency_seconds"]) for item in results]
    return {
        "tasks": total,
        "task_success_rate": _percent(passed, total),
        "first_attempt_success_rate": _percent(first_pass, total),
        "repair_success_rate": _percent(repaired, total),
        "valid_protocol_rate": _percent(protocol_valid, total),
        "patch_application_rate": _percent(patch_applied, total),
        "scope_compliance_rate": _percent(scope_compliant, total),
        "escalation_rate": _percent(escalated, total),
        "median_changed_lines": statistics.median(changed_lines) if changed_lines else None,
        "p50_latency_seconds": statistics.median(latencies) if latencies else None,
        "p95_latency_seconds": (
            sorted(latencies)[max(0, math.ceil(0.95 * len(latencies)) - 1)] if latencies else None
        ),
        "raw_output_evidence_rate": _percent(
            sum(
                1
                for item in results
                if all("raw_output" in attempt for attempt in item["evidence"]["attempts"])
            ),
            total,
        ),
    }


def evaluate_backend(
    backend: GenerationBackend,
    cases_path: Path,
    output_dir: Path,
    *,
    model_id: str,
    model_revision: str,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Run a model on every held-out case and persist raw, reproducible evidence."""
    cases = load_eval_cases(cases_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = NovaRunner(backend, timeout_seconds=timeout_seconds)
    results: list[dict[str, Any]] = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="nova-v12-eval-") as temp:
            workspace = Path(temp) / "workspace"
            workspace.mkdir()
            write_workspace(workspace, case["files_before"])
            started = time.perf_counter()
            evidence = runner.run(case["task"], workspace)
            results.append(
                {
                    "task_id": case["task"].task_id,
                    "provenance": case["provenance"],
                    "latency_seconds": time.perf_counter() - started,
                    "evidence": evidence.to_dict(),
                }
            )

    raw_path = output_dir / "raw_results.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(canonical_json(result) + "\n")

    report = {
        "schema_version": "nova.evaluation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {"id": model_id, "revision": model_revision},
        "cases": {"path": cases_path.name, "sha256": sha256_file(cases_path)},
        "raw_results": {"path": raw_path.name, "sha256": sha256_file(raw_path)},
        "metrics": aggregate_metrics(results),
    }
    report["evaluation_sha256"] = sha256_text(canonical_json(report))
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def load_evaluation(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    supplied = value.get("evaluation_sha256")
    unsigned = dict(value)
    unsigned.pop("evaluation_sha256", None)
    if supplied != sha256_text(canonical_json(unsigned)):
        raise ContractError(f"evaluation hash is invalid: {path}")
    raw = path.parent / value["raw_results"]["path"]
    if sha256_file(raw) != value["raw_results"]["sha256"]:
        raise ContractError(f"raw evaluation evidence hash mismatch: {raw}")
    return value


def release_decision(
    candidate_path: Path,
    baseline_path: Path,
    *,
    minimum_tasks: int = 1_000,
    minimum_protocol_rate: float = 0.99,
    minimum_patch_application_rate: float = 0.97,
    minimum_first_attempt_success_rate: float = 0.70,
    minimum_success_rate: float = 0.80,
    maximum_success_regression: float = 0.02,
) -> dict[str, Any]:
    """Compare candidate with untouched base and enforce locked release gates."""
    candidate = load_evaluation(candidate_path)
    baseline = load_evaluation(baseline_path)
    c = candidate["metrics"]
    b = baseline["metrics"]
    same_cases = candidate["cases"]["sha256"] == baseline["cases"]["sha256"]
    gates = {
        "same_held_out_cases": same_cases,
        "minimum_tasks": c["tasks"] >= minimum_tasks,
        "valid_protocol_rate": c["valid_protocol_rate"] >= minimum_protocol_rate,
        "patch_application_rate": (c["patch_application_rate"] >= minimum_patch_application_rate),
        "first_attempt_success_rate": (
            c["first_attempt_success_rate"] >= minimum_first_attempt_success_rate
        ),
        "atomic_task_success_rate": c["task_success_rate"] >= minimum_success_rate,
        "scope_compliance": c["scope_compliance_rate"] == 1.0,
        "raw_evidence_complete": c["raw_output_evidence_rate"] == 1.0,
        "no_success_regression_vs_base": (
            c["task_success_rate"] >= b["task_success_rate"] - maximum_success_regression
        ),
    }
    report = {
        "schema_version": RELEASE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "evaluation": str(candidate_path),
            "evaluation_sha256": candidate["evaluation_sha256"],
            "model": candidate["model"],
            "metrics": c,
        },
        "baseline": {
            "evaluation": str(baseline_path),
            "evaluation_sha256": baseline["evaluation_sha256"],
            "model": baseline["model"],
            "metrics": b,
        },
        "gates": gates,
        "release_status": "passed" if all(gates.values()) else "blocked",
    }
    report["release_sha256"] = sha256_text(canonical_json(report))
    return report


def main() -> None:  # pragma: no cover - exercised by CLI smoke tests
    parser = argparse.ArgumentParser(description="Evaluate Nova V12 atomic patch execution")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--cases", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--revision", default="unknown")
    run.add_argument("--backend", choices=["ollama", "recorded"], default="ollama")
    run.add_argument("--recorded-outputs", type=Path)
    run.add_argument("--ollama-url", default="http://localhost:11434")
    run.add_argument("--timeout", type=int, default=60)

    release = sub.add_parser("release")
    release.add_argument("--candidate", type=Path, required=True)
    release.add_argument("--baseline", type=Path, required=True)
    release.add_argument("--output", type=Path, required=True)
    release.add_argument("--minimum-tasks", type=int, default=1_000)
    release.add_argument("--minimum-protocol-rate", type=float, default=0.99)
    release.add_argument("--minimum-patch-application-rate", type=float, default=0.97)
    release.add_argument("--minimum-first-attempt-success-rate", type=float, default=0.70)
    release.add_argument("--minimum-success-rate", type=float, default=0.80)
    release.add_argument("--maximum-success-regression", type=float, default=0.02)

    args = parser.parse_args()
    try:
        if args.command == "run":
            if args.backend == "recorded":
                if args.recorded_outputs is None:
                    raise ContractError("--recorded-outputs is required")
                backend: GenerationBackend = RecordedBackend.from_jsonl(args.recorded_outputs)
            else:
                backend = OllamaBackend(args.model, base_url=args.ollama_url)
            result = evaluate_backend(
                backend,
                args.cases,
                args.output_dir,
                model_id=args.model,
                model_revision=args.revision,
                timeout_seconds=args.timeout,
            )
        else:
            result = release_decision(
                args.candidate,
                args.baseline,
                minimum_tasks=args.minimum_tasks,
                minimum_protocol_rate=args.minimum_protocol_rate,
                minimum_patch_application_rate=args.minimum_patch_application_rate,
                minimum_first_attempt_success_rate=(args.minimum_first_attempt_success_rate),
                minimum_success_rate=args.minimum_success_rate,
                maximum_success_regression=args.maximum_success_regression,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except (ContractError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
