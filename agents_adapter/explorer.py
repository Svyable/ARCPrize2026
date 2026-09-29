"""Drop-in agent for the ARC-AGI-3-Agents framework.

Install: copy this file to  ARC-AGI-3-Agents/agents/templates/explorer.py, make the
`arcagent/` package importable (copy it next to `agents/` or `pip install -e .`), and
register the class in `agents/__init__.py` the same way the bundled templates are.
Run:  uv run main.py --agent=explorer

NOTE: written from the documented interface (is_done / choose_action, FrameData,
GameAction, GameState). The framework source was not available when this was
written, so the imports below try both known module layouts; check them first.
"""
from __future__ import annotations

from typing import Any

try:  # newer layout
    from arcengine import FrameData, GameAction, GameState
except ImportError:  # older layout
    from ..structs import FrameData, GameAction, GameState  # type: ignore

from ..agent import Agent  # type: ignore

from arcagent import ExplorerAgent, Move


class Explorer(Agent):
    MAX_ACTIONS = 5000  # framework default is tiny; the scorer, not this cap, is the real limit

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.brain = ExplorerAgent(seed=0, max_actions=self.MAX_ACTIONS)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return self.brain.is_done(frames, latest_frame)

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        move = self.brain.choose_action(frames, latest_frame)
        return _to_game_action(move)


def _to_game_action(move: Move) -> GameAction:
    action = GameAction.RESET if move.action_id == 0 else GameAction[f"ACTION{move.action_id}"]
    if move.action_id == 6:
        action.set_data({"x": int(move.x), "y": int(move.y)})
    action.reasoning = move.reason
    return action
