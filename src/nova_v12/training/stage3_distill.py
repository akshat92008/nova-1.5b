from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from nova_v12.eval.scorer import score_result
from nova_v12.schemas import GenerationResult, load_jsonl, write_jsonl


def build_distillation_data(
    result_paths: list[str | Path],
    *,
    sft_output: str | Path,
    dpo_output: str | Path,
    minimum_score: float = 0.99,
) -> tuple[int, int]:
    grouped: dict[str, list[tuple[GenerationResult, float, dict[str, Any]]]] = defaultdict(list)
    for path in result_paths:
        for value in load_jsonl(path):
            result = GenerationResult.from_dict(value)
            score = score_result(result)
            grouped[result.task_id].append((result, score.score, score.to_dict()))

    sft: list[dict[str, Any]] = []
    dpo: list[dict[str, Any]] = []
    for task_id, candidates in grouped.items():
        candidates.sort(key=lambda item: item[1], reverse=True)
        chosen, chosen_score, chosen_evidence = candidates[0]
        if chosen_score < minimum_score or not chosen.task:
            continue
        task = chosen.task
        prompt = str(task.get("prompt", ""))
        mode = {
            "code_generation": "code",
            "fim": "fim",
            "debugging": "debug",
            "repository_editing": "edit",
            "tool_use": "agent",
            "instruction_following": "code",
        }.get(chosen.category, "code")
        sft.append(
            {
                "id": f"distill-{task_id}",
                "mode": mode,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen.raw_output},
                ],
                "provenance": {
                    "source": "execution-ranked-distillation",
                    "candidate": chosen.candidate_id,
                },
                "verification": {"passed": True, "score": chosen_score, "details": chosen_evidence},
            }
        )
        if len(candidates) > 1:
            rejected, rejected_score, rejected_evidence = candidates[-1]
            if rejected.raw_output != chosen.raw_output and rejected_score < chosen_score:
                dpo.append(
                    {
                        "id": f"distill-pair-{task_id}",
                        "prompt": prompt,
                        "chosen": chosen.raw_output,
                        "rejected": rejected.raw_output,
                        "chosen_evidence": {
                            "passed": True,
                            "score": chosen_score,
                            "details": chosen_evidence,
                        },
                        "rejected_evidence": {
                            "passed": rejected_score >= minimum_score,
                            "score": rejected_score,
                            "details": rejected_evidence,
                        },
                        "repository_snapshot": str(
                            task.get("metadata", {}).get("repository_snapshot", task_id)
                        ),
                    }
                )
    write_jsonl(sft_output, sft)
    write_jsonl(dpo_output, dpo)
    return len(sft), len(dpo)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--sft-output", required=True)
    parser.add_argument("--dpo-output", required=True)
    parser.add_argument("--minimum-score", type=float, default=0.99)
    args = parser.parse_args(argv)
    sft, dpo = build_distillation_data(
        args.results,
        sft_output=args.sft_output,
        dpo_output=args.dpo_output,
        minimum_score=args.minimum_score,
    )
    print({"sft": sft, "dpo": dpo})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
