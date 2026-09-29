"""Dev-only: BFS shortest solution for level 1 of a game, using simple actions + top-K click candidates."""
import copy, hashlib, logging, sys, collections, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1])); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from oracle import do, skey
from arcagent.vision import click_candidates
logging.disable(logging.CRITICAL)

def run(gid, tlimit=200, K=16, lvl=0):
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files", logger=logging.getLogger("q"))
    env = arc.make(gid); g0 = env._game; fr = env.observation_space
    base = next(e for e in arc.get_environments() if e.game_id.startswith(gid)).baseline_actions
    simple = [a for a in fr.available_actions if a not in (0, 6)]
    click = 6 in fr.available_actions
    lvl0 = g0._score
    q = collections.deque([(g0, [], fr)]); seen = {skey(g0, fr)}; t0 = time.time(); n = 0
    while q:
        g, path, fr = q.popleft()
        acts = [(a, 0, 0) for a in simple]
        if click: acts += [(6, x, y) for x, y in click_candidates(fr.frame[-1])[:K]]
        for a in acts:
            g2 = copy.deepcopy(g); f2 = do(g2, a); n += 1
            if f2.state is GameState.GAME_OVER: continue
            if f2.levels_completed > lvl0:
                return f"{gid} SOLVED depth {len(path)+1} baseline {base[lvl0]} nodes {n} seq {[(x[0],x[1],x[2]) if x[0]==6 else x[0] for x in path+[a]]}"
            k = skey(g2, f2)
            if k in seen: continue
            seen.add(k); q.append((g2, path + [a], f2))
        if time.time() - t0 > tlimit: return f"{gid} TIMEOUT nodes {n} depth {len(path)} frontier {len(q)} baseline {base[lvl0]}"
    return f"{gid} EXHAUSTED nodes {n} baseline {base[lvl0]}"

if __name__ == "__main__":
    print(run(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 200), flush=True)
