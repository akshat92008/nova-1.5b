from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from nova_v12.eval.backends import OllamaBackend, TransformersBackend
from nova_v12.eval.runner import run_tasks
from nova_v12.eval.scorer import score_results


def run_bakeoff(
    config_path: str | Path,
    task_path: str | Path,
    output_dir: str | Path,
    *,
    track: str = "foundation_track",
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    candidates = config.get(track)
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"candidate track is empty or missing: {track}")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    combined: dict[str, Any] = {}
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        model = str(candidate["model"])
        if model.startswith("REPLACE_WITH_"):
            raise ValueError(f"candidate {candidate_id} still contains a placeholder model")
        backend_name = str(candidate.get("backend", "transformers"))
        if backend_name == "ollama":
            backend = OllamaBackend(model, base_url=candidate.get("base_url"))
        elif backend_name == "transformers":
            backend = TransformersBackend(
                model,
                trust_remote_code=bool(candidate.get("trust_remote_code", False)),
                revision=candidate.get("revision"),
            )
        else:
            raise ValueError(f"unsupported backend: {backend_name}")
        candidate_dir = root / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        results_path = candidate_dir / "results.jsonl"
        run_tasks(
            backend,
            candidate_id,
            task_path,
            results_path,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        scores, summary = score_results(results_path)
        report = {"summary": summary, "scores": [item.to_dict() for item in scores]}
        (candidate_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        combined[candidate_id] = summary.get(candidate_id, {})
    leaderboard = sorted(
        ({"candidate": candidate_id, **metrics} for candidate_id, metrics in combined.items()),
        key=lambda item: float(item.get("overall", 0.0)),
        reverse=True,
    )
    output = {"track": track, "leaderboard": leaderboard}
    (root / "leaderboard.json").write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output
