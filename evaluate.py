"""Play the public games locally and score them like the competition toolkit.

    python evaluate.py [--games ls20,ft09] [--max-actions 2000] [--agent explorer|random]

Score per level = min((baseline/actions)^2, 1.15)*100, weighted by 1-based level index;
unfinished levels score 0 (mirrors arc_agi.scorecard). Every env.step, RESET included, counts.
"""
from __future__ import annotations

import argparse
import logging
import random
import time
from concurrent.futures import ProcessPoolExecutor

ENV_DIR = "environment_files"


def play_game(gid: str, agent_name: str, max_actions: int, seed: int, opts: dict | None = None) -> dict:
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction, GameState

    from arcagent import ExplorerAgent
    from arcagent.gym import to_game_action

    logging.disable(logging.CRITICAL)
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ENV_DIR, logger=logging.getLogger("q"))
    info = next(e for e in arc.get_environments() if e.game_id.startswith(gid))
    env = arc.make(gid)
    agent = ExplorerAgent(seed=seed, max_actions=max_actions, **(opts or {})) if agent_name == "explorer" else None
    rng = random.Random(seed)
    frame = env.observation_space
    level_actions: dict[int, int] = {}
    n, t0, resets = 0, time.time(), 0
    err = None
    while n < max_actions and frame.state is not GameState.WIN:
        lvl = frame.levels_completed
        if agent is not None:
            move = agent.choose_action([], frame)
            action = to_game_action(move)
        else:
            if frame.state in (GameState.GAME_OVER, GameState.NOT_PLAYED):
                action = GameAction.RESET
            else:
                action = GameAction.from_id(rng.choice([a for a in frame.available_actions if a != 0] or [1]))
                if action.is_complex():
                    action.set_data({"x": rng.randrange(64), "y": rng.randrange(64)})
        resets += action is GameAction.RESET
        data = action.action_data.model_dump()
        nxt = env.step(action, data=data, reasoning={})
        if nxt is None:
            err = "env returned None"
            break
        frame = nxt
        n += 1
        level_actions[lvl] = level_actions.get(lvl, 0) + 1
    base = info.baseline_actions
    scores, weights = [], []
    for i, b in enumerate(base):
        done = i < frame.levels_completed
        a = level_actions.get(i, 0)
        scores.append(min((b / a) ** 2, 1.15) * 100 if done and a else 0.0)
        weights.append(i + 1)
    score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    score = min(score, sum(w for s, w in zip(scores, weights) if s > 0) / sum(weights) * 100)  # toolkit's cap
    return {
        "game": gid, "state": frame.state.name, "levels": f"{frame.levels_completed}/{len(base)}",
        "actions": n, "resets": resets, "score": round(score, 2), "secs": round(time.time() - t0, 1),
        "errors": getattr(agent, "errors", 0), "level_actions": level_actions, "err": err,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="")
    ap.add_argument("--max-actions", type=int, default=2000)
    ap.add_argument("--agent", default="explorer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--opts", default="", help="ExplorerAgent kwargs, e.g. momentum=0,min_rate=0.3")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    from pathlib import Path

    gids = a.games.split(",") if a.games else sorted(p.name for p in Path(ENV_DIR).iterdir() if p.is_dir())
    with ProcessPoolExecutor(a.workers) as ex:
        opts = {k: (float(v) if "." in v else int(v)) for k, v in (kv.split("=") for kv in a.opts.split(",") if kv)}
        opts = {k: bool(v) if k in ("momentum", "novelty_rate") else v for k, v in opts.items()}
        futs = [ex.submit(play_game, g, a.agent, a.max_actions, a.seed, opts) for g in gids]
        rows = [f.result() for f in futs]
    tot = 0.0
    for r in rows:
        tot += r["score"]
        if not a.quiet: print(f"{r['game']:5s} {r['state']:12s} levels={r['levels']:5s} actions={r['actions']:5d} resets={r['resets']:4d} "
              f"score={r['score']:6.2f} {r['secs']:5.1f}s err={r['errors']} {r['err'] or ''}")
    lv = sum(int(r["levels"].split("/")[0]) for r in rows)
    print(f"MEAN SCORE over {len(rows)} games: {tot / len(rows):.2f}   levels solved: {lv}   opts={a.opts or 'default'}")


if __name__ == "__main__":
    main()
