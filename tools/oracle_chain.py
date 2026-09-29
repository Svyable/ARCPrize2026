"""Dev-only: chain BFS solutions across consecutive levels of one game."""
import copy, logging, sys, collections, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1])); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from arc_agi import Arcade, OperationMode
from arcengine import GameState
from oracle import do, skey
from arcagent.vision import click_candidates
logging.disable(logging.CRITICAL)

def solve_level(g0, fr, simple, click, tlimit, K):
    lvl0 = g0._score
    q = collections.deque([(g0, [], fr)]); seen = {skey(g0, fr)}; t0 = time.time(); n = 0
    while q:
        g, path, fr = q.popleft()
        acts = [(a, 0, 0) for a in simple]
        if click: acts += [(6, x, y) for x, y in click_candidates(fr.frame[-1])[:K]]
        for a in acts:
            g2 = copy.deepcopy(g); f2 = do(g2, a); n += 1
            if f2.state is GameState.GAME_OVER: continue
            if f2.levels_completed > lvl0: return path + [a], g2, f2, n
            k = skey(g2, f2)
            if k in seen: continue
            seen.add(k); q.append((g2, path + [a], f2))
        if time.time() - t0 > tlimit: return None, None, None, n
    return None, None, None, n

gid = sys.argv[1]; nlev = int(sys.argv[2]); tl = float(sys.argv[3]); K = int(sys.argv[4]) if len(sys.argv) > 4 else 16
arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files", logger=logging.getLogger("q"))
env = arc.make(gid); g = env._game; fr = env.observation_space
base = next(e for e in arc.get_environments() if e.game_id.startswith(gid)).baseline_actions
simple = [a for a in fr.available_actions if a not in (0, 6)]; click = 6 in fr.available_actions
for lv in range(nlev):
    sol, g2, f2, n = solve_level(g, fr, simple, click, tl, K)
    if sol is None: print(gid, "L%d" % lv, "unsolved in", tl, "s; nodes", n, "baseline", base[lv], flush=True); break
    print(gid, "L%d" % lv, "opt", len(sol), "baseline", base[lv], "nodes", n, "seq", [(a[1], a[2]) if a[0] == 6 else a[0] for a in sol], flush=True)
    g, fr = g2, f2
