"""Execution-backed foundation selection and immutable lock generation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import canonical_json, sha256_text
from .evaluation import load_evaluation
from .schema import ContractError

LOCK_SCHEMA = "nova.foundation-lock.v1"


def score_evaluation(report: dict[str, Any]) -> float:
    """Score only measured V12 product metrics; no style heuristics."""
    metrics = report["metrics"]
    return (
        0.45 * metrics["task_success_rate"]
        + 0.20 * metrics["valid_protocol_rate"]
        + 0.15 * metrics["patch_application_rate"]
        + 0.10 * metrics["scope_compliance_rate"]
        + 0.10 * metrics["first_attempt_success_rate"]
    )


def select_foundation(
    evaluations: list[Path],
    metadata_path: Path,
    *,
    minimum_tasks: int = 100,
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidates = metadata.get("candidates")
    if not isinstance(candidates, dict):
        raise ContractError("foundation metadata must contain a candidates object")

    reports = [load_evaluation(path) for path in evaluations]
    if len(reports) < 2:
        raise ContractError("foundation bake-off requires at least two candidates")
    case_hashes = {report["cases"]["sha256"] for report in reports}
    if len(case_hashes) != 1:
        raise ContractError("foundation candidates were not evaluated on identical cases")
    if any(report["metrics"]["tasks"] < minimum_tasks for report in reports):
        raise ContractError(f"every foundation evaluation needs at least {minimum_tasks} tasks")

    ranked: list[dict[str, Any]] = []
    for path, report in zip(evaluations, reports):
        model_id = report["model"]["id"]
        candidate = candidates.get(model_id)
        if not isinstance(candidate, dict):
            raise ContractError(f"missing candidate metadata for {model_id}")
        required = {"hf_id", "revision", "licence"}
        missing = required - candidate.keys()
        if missing:
            raise ContractError(f"{model_id} metadata missing {sorted(missing)}")
        if report["metrics"]["scope_compliance_rate"] < 1.0:
            raise ContractError(f"{model_id} failed scope-compliance safety gate")
        ranked.append(
            {
                "candidate_id": model_id,
                "score": score_evaluation(report),
                "evaluation_path": str(path),
                "evaluation_sha256": report["evaluation_sha256"],
                "metrics": report["metrics"],
                **candidate,
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["candidate_id"]))
    winner = ranked[0]
    return {
        "schema_version": LOCK_SCHEMA,
        "decision_status": "locked",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": winner["candidate_id"],
        "hf_id": winner["hf_id"],
        "revision": winner["revision"],
        "licence": winner["licence"],
        "evaluation_sha256": winner["evaluation_sha256"],
        "held_out_cases_sha256": next(iter(case_hashes)),
        "selection_score": winner["score"],
        "ranking": [
            {
                "candidate_id": item["candidate_id"],
                "score": item["score"],
                "evaluation_sha256": item["evaluation_sha256"],
                "metrics": item["metrics"],
            }
            for item in ranked
        ],
        "selection_policy": {
            "execution_only": True,
            "minimum_tasks": minimum_tasks,
            "weights": {
                "task_success_rate": 0.45,
                "valid_protocol_rate": 0.20,
                "patch_application_rate": 0.15,
                "scope_compliance_rate": 0.10,
                "first_attempt_success_rate": 0.10,
            },
        },
    }


def main() -> None:  # pragma: no cover - exercised by CLI smoke tests
    parser = argparse.ArgumentParser(description="Lock Nova V12 foundation selection")
    parser.add_argument("--evaluation", action="append", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-tasks", type=int, default=100)
    args = parser.parse_args()
    try:
        lock = select_foundation(
            args.evaluation,
            args.metadata,
            minimum_tasks=args.minimum_tasks,
        )
    except (ContractError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    lock["lock_sha256"] = sha256_text(canonical_json(lock))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
