"""Dev-only oracle: BFS on deep copies of the true engine to find shortest solutions per level."""
import copy, hashlib, pickle, logging, sys, collections, time
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState, ActionInput
logging.disable(logging.CRITICAL)

FULL = False

def skey(g, fr):
    parts = [fr.frame[-1].tobytes()]
    for sp in g.current_level.get_sprites():
        parts.append(repr((sp.name, sp.x, sp.y, getattr(sp, 'rotation', 0), getattr(sp, 'is_visible', 1), getattr(sp, 'interaction', 0))).encode())
        parts.append(sp.pixels.tobytes())
    return hashlib.md5(b"|".join(parts)).digest()

def do(g, a):
    ai = ActionInput(id=GameAction.from_id(a[0]), data=({"x": a[1], "y": a[2]} if a[0] == 6 else {}))
    return g.perform_action(ai, raw=True)

def bfs(game, actions, max_nodes=3000000, maxdepth=400, tlimit=1500):
    lvl0 = do(game, (0, 0, 0)).levels_completed if False else game._score
    seen = {}
    q = collections.deque([(game, [])])
    t0 = time.time(); n = 0
    while q:
        g, path = q.popleft()
        if len(path) >= maxdepth: continue
        for a in actions:
            g2 = copy.deepcopy(g)
            fr = do(g2, a)
            n += 1
            if fr.state is GameState.GAME_OVER: continue
            if fr.levels_completed > lvl0 or fr.state is GameState.WIN:
                return path + [a], n
            k = skey(g2, fr) if FULL else fr.frame[-1].tobytes()
            if k in seen: continue
            seen[k] = 1
            q.append((g2, path + [a]))
            if n > max_nodes or time.time() - t0 > tlimit: return None, n
    return None, n

if __name__ == "__main__":
    FULL = len(sys.argv) > 4
    gid = sys.argv[1]; acts = [(int(x), 0, 0) for x in sys.argv[2].split(",")]
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files", logger=logging.getLogger("q"))
    env = arc.make(gid); g = env._game
    base = next(e for e in arc.get_environments() if e.game_id.startswith(gid)).baseline_actions
    for lvl in range(int(sys.argv[3]) if len(sys.argv) > 3 else 2):
        sol, n = bfs(g, acts)
        print(gid, "level", lvl, "baseline", base[lvl], "oracle-optimal", None if sol is None else len(sol), "nodes", n, flush=True)
        if sol is None: break
        for a in sol: do(g, a)
