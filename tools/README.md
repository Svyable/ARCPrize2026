# Dev tools (not used by the agent at run time)

Run from the repo root inside a Python >= 3.12 venv with `arc-agi`, `arcengine`, `numpy`.

| script | purpose |
|---|---|
| `compare.py N cfg1 cfg2 ...` | play all public games for N actions per config (`k=v,k=v` ExplorerAgent kwargs) and print per-game solved levels + actions |
| `profile_all.py` | per-game graph stats (nodes, untried, deaths, HUD mask, life budget) |
| `oracle.py`, `oracle_all.py`, `oracle_chain.py` | **use engine internals** (deep-copying the game) to BFS the true shortest solution of a level, e.g. `python tools/oracle_all.py vc33 240`. Used to learn how long winning sequences really are (3-18 actions) versus human baselines; useless for private games. |
| `show.py game...` | ASCII-render a game's first frame |
