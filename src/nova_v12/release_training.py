"""Reproducible, gate-driven training orchestration for Nova V12."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .dataset import canonical_json, sha256_file, sha256_text, validate_manifest
from .schema import ContractError

FOUNDATION_LOCK_SCHEMA = "nova.foundation-lock.v1"
RUN_SCHEMA = "nova.training-run.v1"


@dataclass(frozen=True)
class Stage:
    name: str
    entry_gate: str
    command: tuple[str, ...]


def load_foundation_lock(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(
            f"missing foundation lock: {path}; run the execution-based bake-off first"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != FOUNDATION_LOCK_SCHEMA:
        raise ContractError("unsupported foundation lock schema")
    if value.get("decision_status") != "locked":
        raise ContractError("foundation decision is not locked")
    supplied_hash = value.get("lock_sha256")
    unsigned = dict(value)
    unsigned.pop("lock_sha256", None)
    if supplied_hash != sha256_text(canonical_json(unsigned)):
        raise ContractError("foundation lock hash is invalid")
    required = {"candidate_id", "hf_id", "revision", "licence", "evaluation_sha256"}
    missing = required - value.keys()
    if missing:
        raise ContractError(f"foundation lock missing keys: {sorted(missing)}")
    if value["licence"].lower() not in {
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
    }:
        raise ContractError("foundation licence is not approved")
    return value


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"missing training config: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "nova.training-config.v1":
        raise ContractError("unsupported training config")
    return value


def build_stages(
    config: dict[str, Any],
    foundation: dict[str, Any],
    dataset_dir: Path,
    run_dir: Path,
) -> tuple[Stage, ...]:
    python = str(config.get("python", "python"))
    base = foundation["hf_id"]
    revision = foundation["revision"]
    common = ["--trust-remote-code"] if bool(config["foundation"].get("trust_remote_code")) else []
    stage2 = run_dir / "stage2-sft"
    stage4 = run_dir / "stage4-dpo"
    preference_dir = Path(config["data"]["preference_dir"])
    return (
        Stage(
            name="stage2_sft",
            entry_gate="foundation_lock_and_verified_atomic_patch_data",
            command=tuple(
                [
                    python,
                    "-m",
                    "nova_v12.training.stage2_sft",
                    "--base-model",
                    base,
                    "--revision",
                    revision,
                    "--data-dir",
                    str(dataset_dir),
                    "--output-dir",
                    str(stage2),
                    "--minimum-verified",
                    str(config["data"]["minimum_verified_examples"]),
                    *common,
                ]
            ),
        ),
        Stage(
            name="stage4_dpo",
            entry_gate="sft_release_metrics_non_regressing_and_verified_pairs",
            command=tuple(
                [
                    python,
                    "-m",
                    "nova_v12.training.stage4_dpo",
                    "--sft-adapter",
                    str(stage2),
                    "--data-dir",
                    str(preference_dir),
                    "--output-dir",
                    str(stage4),
                    "--minimum-pairs",
                    str(config["data"]["minimum_preference_pairs"]),
                ]
            ),
        ),
    )


def preflight(
    config_path: Path,
    foundation_lock_path: Path,
    dataset_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    foundation = load_foundation_lock(foundation_lock_path)
    minimum = int(config["data"]["minimum_verified_examples"])
    dataset = validate_manifest(dataset_dir, minimum_verified=minimum)
    from .preference import validate_preference_manifest

    preference_dir = Path(config["data"]["preference_dir"])
    validate_preference_manifest(
        preference_dir,
        minimum_pairs=int(config["data"]["minimum_preference_pairs"]),
    )
    return config, foundation, dataset


def create_run_manifest(
    config_path: Path,
    foundation_lock_path: Path,
    dataset_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    config, foundation, dataset = preflight(
        config_path,
        foundation_lock_path,
        dataset_dir,
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    stages = build_stages(config, foundation, dataset_dir, run_dir)
    value = {
        "schema_version": RUN_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned",
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "foundation_lock": {
            "path": str(foundation_lock_path),
            "sha256": sha256_file(foundation_lock_path),
            "candidate_id": foundation["candidate_id"],
            "hf_id": foundation["hf_id"],
            "revision": foundation["revision"],
        },
        "dataset": {
            "path": str(dataset_dir),
            "manifest_sha256": dataset["manifest_sha256"],
            "verified_examples": dataset["verified_examples"],
        },
        "stages": [
            {
                "name": stage.name,
                "entry_gate": stage.entry_gate,
                "command": list(stage.command),
                "status": "pending",
            }
            for stage in stages
        ],
    }
    value["run_sha256"] = sha256_text(canonical_json(value))
    (run_dir / "run.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return value


def main() -> None:  # pragma: no cover - exercised by CLI smoke tests
    parser = argparse.ArgumentParser(description="Nova V12 training preflight and planner")
    parser.add_argument("--config", type=Path, default=Path("configs/nova-v12.yaml"))
    parser.add_argument("--foundation-lock", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = create_run_manifest(
            args.config,
            args.foundation_lock,
            args.dataset_dir,
            args.run_dir,
        )
    except (ContractError, KeyError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
