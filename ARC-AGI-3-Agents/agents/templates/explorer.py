"""Explorer agent for the ARC-AGI-3-Agents framework (the class the swarm instantiates).

Installed at ARC-AGI-3-Agents/agents/templates/explorer.py and registered in
ARC-AGI-3-Agents/agents/__init__.py; run with `uv run main.py --agent=explorer`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from arcengine import FrameData, GameAction

from ..agent import Agent

try:
    from arcagent import ExplorerAgent
    from arcagent.gym import to_game_action
except ImportError:  # repo layout: <root>/ARC-AGI-3-Agents/agents/templates/explorer.py, package at <root>/arcagent
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from arcagent import ExplorerAgent
    from arcagent.gym import to_game_action


class Explorer(Agent):
    """Graph-exploration agent: see arcagent/core.py."""

    MAX_ACTIONS = 100000  # the scorer/time budget, not this cap, is the real limit

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.brain = ExplorerAgent(seed=0, max_actions=self.MAX_ACTIONS)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return self.brain.is_done(frames, latest_frame)

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        return to_game_action(self.brain.choose_action(frames, latest_frame))
