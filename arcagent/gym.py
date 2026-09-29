"""Glue between Move and the real arcengine types (imported lazily: arcengine needs py>=3.12)."""
from __future__ import annotations

from .core import Move


def to_game_action(move: Move):
    from arcengine import GameAction

    action = GameAction.RESET if move.action_id == 0 else GameAction.from_id(move.action_id)
    if move.action_id == 6:
        action.set_data({"x": int(move.x), "y": int(move.y)})
    action.reasoning = move.reason
    return action
