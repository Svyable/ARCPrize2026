"""Local harness: drives an agent against a mock game the way the framework does,
and scores it with the competition formula (per level min(baseline/actions, 1)^2,
level-index-weighted average)."""
from __future__ import annotations

from .core import ExplorerAgent
from .mock_env import Frame, MockGame


def play(game: MockGame, agent: ExplorerAgent) -> dict:
    latest = Frame([[[0]]], "NOT_PLAYED", 0, list(game.available))
    frames = [latest]
    per_level: dict[int, int] = {}
    total = 0
    while not agent.is_done(frames, latest):
        move = agent.choose_action(frames, latest)
        level_before = game.level
        latest = game.step(move.action_id, move.x, move.y)
        frames.append(latest)
        total += 1
        per_level[level_before] = per_level.get(level_before, 0) + 1
    completed = game.level
    return {"state": game.state, "levels_completed": completed, "actions": total, "per_level": per_level}


def score(result: dict, baselines: list[int]) -> float:
    """Per-game score; unfinished levels score 0."""
    num = den = 0.0
    for i, base in enumerate(baselines):
        w = i + 1
        den += w
        if i < result["levels_completed"]:
            num += w * min(base / max(result["per_level"].get(i, 1), 1), 1.0) ** 2
    return num / den
