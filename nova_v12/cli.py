"""Unified command-line entry point for the Nova V12 system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import build_verified_dataset, validate_manifest
from .runner import NovaRunner, OllamaBackend
from .schema import AtomicTask, ContractError


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(prog="nova-v12")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="execute one verified atomic patch")
    run.add_argument("--task", type=Path, required=True)
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--ollama-url", default="http://localhost:11434")
    run.add_argument("--timeout", type=int, default=60)
    run.add_argument("--evidence", type=Path)

    build = sub.add_parser("build-dataset", help="build execution-verified data")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--timeout", type=int, default=60)
    build.add_argument("--allow-rejections", action="store_true")

    check = sub.add_parser("check-dataset", help="verify dataset hashes and evidence")
    check.add_argument("--dataset-dir", type=Path, required=True)
    check.add_argument("--minimum-verified", type=int, default=1)

    args = parser.parse_args()
    try:
        if args.command == "run":
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
        elif args.command == "build-dataset":
            result = build_verified_dataset(
                args.source,
                args.output_dir,
                timeout_seconds=args.timeout,
                fail_on_rejection=not args.allow_rejections,
            )
        else:
            result = validate_manifest(
                args.dataset_dir,
                minimum_verified=args.minimum_verified,
            )
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
