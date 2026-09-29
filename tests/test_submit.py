"""Integration test: the standalone submission runner against a real local game.

Skipped when the ARC toolkit (needs Python >= 3.12) or the public environment files are absent.
"""
import logging
import os
from pathlib import Path

import pytest

pytest.importorskip("arc_agi")
pytest.importorskip("arcengine")

ENV_DIR = Path(__file__).resolve().parents[1] / "environment_files"
pytestmark = pytest.mark.skipif(not ENV_DIR.exists(), reason="environment_files not present")


def test_play_one_runs_a_real_game_and_scores():
    from arc_agi import Arcade, OperationMode

    from arcagent.submit import play_one

    logging.disable(logging.CRITICAL)
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ENV_DIR), logger=logging.getLogger("t"))
    card = arc.open_scorecard(tags=["test"])
    res = play_one(arc, card, "lp85", max_actions=60, max_seconds=60)
    assert res["error"] is None and res["fallbacks"] == 0
    assert res["actions"] <= 60
    assert res["levels"] >= 1  # lp85 level 1 is solved in ~7 actions
    scorecard = arc.close_scorecard(card)
    assert scorecard.total_levels_completed >= 1 and scorecard.total_actions == res["actions"]


def test_time_budget_stops_the_agent():
    from arc_agi import Arcade, OperationMode

    from arcagent.submit import play_one

    logging.disable(logging.CRITICAL)
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ENV_DIR), logger=logging.getLogger("t"))
    card = arc.open_scorecard(tags=["test"])
    res = play_one(arc, card, "ls20", max_actions=100000, max_seconds=0.5)
    assert res["actions"] < 100000 and res["secs"] < 10
