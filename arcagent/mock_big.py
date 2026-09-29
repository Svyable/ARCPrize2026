import time
import random

import numpy as np
from arcagent import ExplorerAgent
from arcagent.mock_env import MockGame
from arcagent.runner import play

class Big(MockGame):
    available=[6]
    n_levels=1
    def __init__(s, njunk=60, seed=1):
        super().__init__(); s.optimal=[8]; s.njunk=njunk; s.seed=seed
    def _load(s,l):
        r=random.Random(s.seed); s.tiles={}
        while len(s.tiles)<8: s.tiles[(r.randrange(2,60,4),r.randrange(2,60,4))]=False
        s.junk=[(r.randrange(0,62),r.randrange(0,62)) for _ in range(s.njunk)]
    def _apply(s,a,x,y):
        for (tx,ty) in s.tiles:
            if tx<=x<=tx+1 and ty<=y<=ty+1: s.tiles[(tx,ty)]=True
        return "win" if all(s.tiles.values()) else None
    def _render(s):
        g=np.zeros((64,64),np.uint8)
        for i,(x,y) in enumerate(s.junk): g[y,x]=4+i%3
        for (tx,ty),d in s.tiles.items(): g[ty:ty+2,tx:tx+2]=3 if d else 2
        return g
if __name__ == "__main__":
    for nj in (0,60,150):
        t=time.time(); a=ExplorerAgent(max_actions=3000); r=play(Big(nj),a); dt=time.time()-t
        print(nj, r["state"], r["actions"], f"{dt/r['actions']*1000:.1f} ms/action", len(a.edges),"nodes")
