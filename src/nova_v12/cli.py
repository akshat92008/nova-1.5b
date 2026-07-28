"""Unified command-line entry point for the Nova V12 system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset import build_verified_dataset, validate_manifest
from .eval.tasks import validate_tasks
from .runner import NovaRunner, OllamaBackend
from .schema import AtomicTask, ContractError
from .schemas import load_jsonl, write_jsonl


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain an object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nova-v12",
        description="Verified Nova V12 runtime and model-development toolkit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("run", help="Execute one verified atomic patch")
    command.add_argument("--task", type=Path, required=True)
    command.add_argument("--workspace", type=Path, required=True)
    command.add_argument("--model", required=True)
    command.add_argument("--ollama-url", default="http://localhost:11434")
    command.add_argument("--timeout", type=int, default=60)
    command.add_argument("--evidence", type=Path)

    command = sub.add_parser(
        "build-dataset",
        help="Build execution-verified atomic patch data",
    )
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--timeout", type=int, default=60)
    command.add_argument("--allow-rejections", action="store_true")

    command = sub.add_parser(
        "check-dataset",
        help="Verify atomic dataset hashes and execution evidence",
    )
    command.add_argument("--dataset-dir", type=Path, required=True)
    command.add_argument("--minimum-verified", type=int, default=1)

    command = sub.add_parser("validate-tasks", help="Validate broad bake-off tasks")
    command.add_argument("path")

    command = sub.add_parser("run-eval", help="Generate broad bake-off outputs")
    command.add_argument("--tasks", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--candidate-id", required=True)
    command.add_argument("--backend", choices=["ollama", "transformers"], required=True)
    command.add_argument("--model", required=True)
    command.add_argument("--revision")
    command.add_argument("--trust-remote-code", action="store_true")
    command.add_argument("--max-tokens", type=int, default=2048)
    command.add_argument("--temperature", type=float, default=0.0)

    command = sub.add_parser("bakeoff", help="Run and score a candidate track")
    command.add_argument("--config", required=True)
    command.add_argument("--tasks", required=True)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--track", default="foundation_track")
    command.add_argument("--max-tokens", type=int, default=2048)
    command.add_argument("--temperature", type=float, default=0.0)

    command = sub.add_parser(
        "score-results",
        help="Execute and score saved broad-bake-off generations",
    )
    command.add_argument("--results", required=True)
    command.add_argument("--output", required=True)

    command = sub.add_parser("split-data", help="Split JSONL by repository or snapshot")
    command.add_argument("--input", required=True)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--key-field", default="repository")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--train", type=float, default=0.98)
    command.add_argument("--validation", type=float, default=0.01)
    command.add_argument("--test", type=float, default=0.01)

    command = sub.add_parser("build-data", help="Build a filtered, auditable code corpus")
    command.add_argument("--config", required=True)

    command = sub.add_parser("scan-contamination", help="Scan JSONL for benchmark leakage")
    command.add_argument("--input", required=True)
    command.add_argument("--signatures", required=True)
    command.add_argument("--output", required=True)

    command = sub.add_parser(
        "verify-mutations",
        help="Keep only execution-verified synthetic mutations",
    )
    command.add_argument("--input", required=True)
    command.add_argument("--output", required=True)

    for name, validator_help in (
        ("validate-sft", "Validate generic SFT JSONL records"),
        ("validate-dpo", "Validate generic DPO JSONL records"),
    ):
        command = sub.add_parser(name, help=validator_help)
        command.add_argument("--input", required=True)
        command.add_argument("--rejected")
    return parser


def _run_atomic(args: argparse.Namespace) -> dict[str, Any]:
    task = AtomicTask.from_dict(_load_json(args.task))
    evidence = NovaRunner(
        OllamaBackend(args.model, base_url=args.ollama_url),
        timeout_seconds=args.timeout,
    ).run(task, args.workspace)
    result = evidence.to_dict()
    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            print(json.dumps(_run_atomic(args), indent=2, sort_keys=True))
            return 0
        if args.command == "build-dataset":
            result = build_verified_dataset(
                args.source,
                args.output_dir,
                timeout_seconds=args.timeout,
                fail_on_rejection=not args.allow_rejections,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "check-dataset":
            result = validate_manifest(
                args.dataset_dir,
                minimum_verified=args.minimum_verified,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "validate-tasks":
            count, errors = validate_tasks(args.path)
            print(json.dumps({"tasks": count, "valid": not errors, "errors": errors}, indent=2))
            return 0 if not errors else 1
        if args.command == "run-eval":
            return _run_broad_eval(args)
        if args.command == "bakeoff":
            from .eval.bakeoff import run_bakeoff

            report = run_bakeoff(
                args.config,
                args.tasks,
                args.output_dir,
                track=args.track,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "score-results":
            from .eval.scorer import score_results

            scores, summary = score_results(args.results)
            payload = {"summary": summary, "scores": [score.to_dict() for score in scores]}
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if args.command == "split-data":
            from .data.split import split_jsonl

            counts = split_jsonl(
                args.input,
                args.output_dir,
                key_field=args.key_field,
                seed=args.seed,
                train=args.train,
                validation=args.validation,
                test=args.test,
            )
            print(json.dumps(counts, indent=2, sort_keys=True))
            return 0
        if args.command == "build-data":
            from .data.pipeline import build_data

            print(json.dumps(build_data(args.config), indent=2, sort_keys=True))
            return 0
        if args.command == "scan-contamination":
            return _scan_contamination(args)
        if args.command == "verify-mutations":
            from .data.mutations import verify_mutation_file

            accepted, rejected = verify_mutation_file(args.input, args.output)
            print(json.dumps({"accepted": accepted, "rejected": rejected}, indent=2))
            return 0 if accepted else 1
        if args.command in {"validate-sft", "validate-dpo"}:
            return _validate_training_records(args)
    except (ContractError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser = build_parser()
        parser.exit(2, f"error: {exc}\n")
    raise AssertionError(args.command)


def _run_broad_eval(args: argparse.Namespace) -> int:
    from .eval.backends import OllamaBackend as EvalOllamaBackend
    from .eval.backends import TransformersBackend
    from .eval.runner import run_tasks

    if args.backend == "ollama":
        backend = EvalOllamaBackend(args.model)
    else:
        backend = TransformersBackend(
            args.model,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
        )
    results = run_tasks(
        backend,
        args.candidate_id,
        args.tasks,
        args.output,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(json.dumps({"generated": len(results), "output": args.output}, indent=2))
    return 0


def _scan_contamination(args: argparse.Namespace) -> int:
    from .data.contamination import ContaminationScanner

    scanner = ContaminationScanner.from_file(args.signatures)
    values = []
    contaminated = 0
    for record in load_jsonl(args.input):
        findings = scanner.scan(record)
        if findings:
            contaminated += 1
        values.append(
            {
                "record": record,
                "findings": [item.to_dict() for item in findings],
            }
        )
    write_jsonl(args.output, values)
    print(json.dumps({"records": len(values), "contaminated": contaminated}, indent=2))
    return 1 if contaminated else 0


def _validate_training_records(args: argparse.Namespace) -> int:
    from .data.validators import validate_dpo_record, validate_sft_record

    validator = validate_sft_record if args.command == "validate-sft" else validate_dpo_record
    accepted = []
    rejected = []
    for record in load_jsonl(args.input):
        report = validator(record)
        if report.valid:
            accepted.append(record)
        else:
            rejected.append({"record": record, "errors": report.errors})
    if args.rejected:
        write_jsonl(args.rejected, rejected)
    print(json.dumps({"valid": len(accepted), "invalid": len(rejected)}, indent=2))
    return 0 if not rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
