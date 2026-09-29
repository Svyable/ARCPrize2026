import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from evaluate import play_game
def parse(o): return {k: float(v) if "." in v else int(v) for k, v in (kv.split("=") for kv in o.split(",") if kv)}
cfgs = sys.argv[2:]; N = int(sys.argv[1])
gids = sorted(p.name for p in Path("environment_files").iterdir())
res = {}
with ProcessPoolExecutor(8) as ex:
    for c in cfgs:
        o = parse(c)
        for k in ("momentum", "novelty_rate"):
            if k in o: o[k] = bool(o[k])
        res[c] = list(ex.map(play_game, gids, ["explorer"]*len(gids), [N]*len(gids), [0]*len(gids), [o]*len(gids)))
print(f"{'game':5s}", *[f"{c[:28]:30s}" for c in cfgs])
for i, g in enumerate(gids):
    cells = []
    for c in cfgs:
        r = res[c][i]; n = int(r["levels"].split("/")[0])
        cells.append(f"{n}:" + ",".join(str(r["level_actions"].get(l, 0)) for l in range(n)))
    print(f"{g:5s}", *[f"{x:30s}" for x in cells])
for c in cfgs:
    print(c, "score", round(sum(r["score"] for r in res[c]) / len(gids), 2), "levels", sum(int(r["levels"].split("/")[0]) for r in res[c]))
