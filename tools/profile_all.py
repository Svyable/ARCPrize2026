import logging, sys, collections
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from concurrent.futures import ProcessPoolExecutor
def prof(gid, N=2000):
    from arc_agi import Arcade, OperationMode
    from arcengine import GameState
    from arcagent import ExplorerAgent
    from arcagent.gym import to_game_action
    logging.disable(logging.CRITICAL)
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files", logger=logging.getLogger("q"))
    env = arc.make(gid); a = ExplorerAgent(max_actions=N); f = env.observation_space
    n = ch = deaths = 0; prev = f.frame[-1]; lv_at = []; clicks = 0
    while n < N and f.state is not GameState.WIN:
        m = a.choose_action([], f); act = to_game_action(m)
        f = env.step(act, data=act.action_data.model_dump(), reasoning={}); n += 1
        clicks += m.action_id == 6
        cur = f.frame[-1]
        ch += (cur.shape != prev.shape) or bool((cur != prev).any()); prev = cur
        deaths += f.state is GameState.GAME_OVER
        if len(lv_at) < f.levels_completed: lv_at.append(n)
    return (gid, f.levels_completed, len(a.edges), sum(len(u) for u in a.untried), len(a.raws), deaths,
            round(ch / n, 2), round(clicks / n, 2), None if a.mask is None else int(a.mask.sum()), a.budget, lv_at[:3])
if __name__ == "__main__":
    from pathlib import Path
    gids = sorted(p.name for p in Path("environment_files").iterdir())
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(prof, gids))
    print("game lv nodes untried raws deaths chg% click% mask budget lvl_at")
    for r in rows: print(*r)
