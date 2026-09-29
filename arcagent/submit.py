"""Standalone competition runner: plays every environment the ARC-AGI toolkit exposes.

Depends only on arc_agi + arcengine + numpy (not the ARC-AGI-3-Agents `agents` package, whose
LLM templates pull in langchain/smolagents). Environment selection follows the toolkit's own
config: OPERATION_MODE (offline|online|competition|normal), ARC_API_KEY, ENVIRONMENTS_DIR.

    python -m arcagent.submit [--games ls20,ft09] [--max-actions 5000] [--time-budget 3600]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from .core import ExplorerAgent
from .gym import to_game_action

log = logging.getLogger("arcagent.submit")


def play_one(arc, card_id: str, game_id: str, max_actions: int, max_seconds: float) -> dict:
    from arcengine import GameState

    t0 = time.monotonic()
    out = {"game": game_id, "actions": 0, "levels": 0, "state": "?", "error": None}
    try:
        env = arc.make(game_id, scorecard_id=card_id)
        if env is None:
            out["error"] = "make() returned None"
            return out
        agent = ExplorerAgent(seed=0, max_actions=max_actions, max_seconds=max_seconds)
        frame = env.observation_space
        while not agent.is_done([], frame) and out["actions"] < max_actions:
            action = to_game_action(agent.choose_action([], frame))
            nxt = env.step(action, data=action.action_data.model_dump(), reasoning={})
            if nxt is None:  # transient failure: don't spin forever
                out["error"] = "step returned None"
                break
            frame = nxt
            out["actions"] += 1
            if frame.state is GameState.WIN:
                break
        out["levels"], out["state"] = frame.levels_completed, frame.state.name
        out["fallbacks"] = agent.errors
    except Exception as e:  # one broken game must not sink the whole run
        log.exception("game %s failed", game_id)
        out["error"] = repr(e)
    out["secs"] = round(time.monotonic() - t0, 1)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="", help="comma-separated game id prefixes (default: all)")
    ap.add_argument("--max-actions", type=int, default=5000, help="per-game action cap")
    ap.add_argument("--time-budget", type=float, default=0, help="total seconds for the whole run (0 = unlimited)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tags", default="explorer")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from arc_agi import Arcade

    arc = Arcade()
    ids = sorted({e.game_id for e in arc.get_environments()})
    if a.games:
        wanted = a.games.split(",")
        ids = [g for g in ids if any(g.startswith(w) for w in wanted)]
    log.info("playing %d games: %s", len(ids), ids)
    card_id = arc.open_scorecard(tags=a.tags.split(","))
    start = time.monotonic()
    done = 0

    def budget_for_next() -> float:
        if not a.time_budget:
            return float("inf")
        left = a.time_budget - (time.monotonic() - start)
        return max(left / max(len(ids) - done, 1) * a.workers, 1.0)

    results = []
    try:
        if a.workers <= 1:
            for g in ids:
                results.append(play_one(arc, card_id, g, a.max_actions, budget_for_next()))
                done += 1
                log.info("%s", results[-1])
        else:
            with ThreadPoolExecutor(a.workers) as ex:
                futs = []
                for g in ids:
                    futs.append(ex.submit(play_one, arc, card_id, g, a.max_actions, budget_for_next()))
                for f in futs:
                    results.append(f.result())
                    log.info("%s", results[-1])
    finally:
        card = arc.close_scorecard(card_id)
        if card is not None:
            log.info("SCORECARD %s", json.dumps(card.model_dump(), default=str)[:4000])
            print(f"levels {card.total_levels_completed}/{card.total_levels}  actions {card.total_actions}  "
                  f"score {getattr(card, 'score', None)}")


if __name__ == "__main__":
    main()
