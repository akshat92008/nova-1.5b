from __future__ import annotations

from nova_v12.schemas import TaskScore


def reward_from_score(score: TaskScore) -> float:
    """Stable bounded reward derived only from executable checks."""
    if score.error:
        return -1.0
    reward = 2.0 * score.score - 1.0
    for check in score.checks:
        if (
            check.name in {"authorised_files_only", "patch_applies", "tests_pass"}
            and not check.passed
        ):
            reward -= 0.25
    return max(-1.0, min(1.0, reward))
