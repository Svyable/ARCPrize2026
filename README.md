# ARC Prize 2026 – ARC-AGI-3 agent

A framework-independent exploration agent (`arcagent/`) plus a thin adapter for the
ARC-AGI-3-Agents framework (`agents_adapter/explorer.py`).

## How it plays
Per level the environment is treated as a deterministic state graph:

* **Nodes** = frames. Volatile HUD cells (step bars, timers) are detected — a row/column/cell
  that changes for *every* action id tried, including bumps and no-ops — and masked out, so
  a shrinking energy bar doesn't make every frame look new.
* **Edges** = (state, action) results. Each pair is tried once. Actions are ranked by how often
  their *kind* (action id, or click-on-colour) changed the frame; ACTION6 targets are connected
  components (rare colours / small objects first) plus a coarse lattice.
* **Navigation**: when the current state is exhausted, BFS along known edges to the nearest state
  with untried actions (no wasted moves re-testing known ground).
* **GAME_OVER**: the fatal edge is recorded, the agent RESETs; the graph persists, so paths
  avoid known deaths.
* **Level up** (score increases): fresh graph, action-kind statistics carried over.
* Any internal exception degrades to a random legal action instead of aborting the run.

## Use
```
pip install -e .[dev]
python -m pytest            # unit + mock-game tests
python run_local.py         # mock games with the competition scoring formula
```
Framework: copy `agents_adapter/explorer.py` to `ARC-AGI-3-Agents/agents/templates/`, register it,
`uv run main.py --agent=explorer`.

## Status / honesty notes
* The competition files (`ARC-AGI-3-Agents/`, `environment_files/`, wheels) were **not available**
  when this was built; nothing has been run against a real ARC-AGI-3 game. The mock games in
  `arcagent/mock_env.py` only check that the mechanics of the agent work.
* `agents_adapter/explorer.py` was written from the documented interface and is unverified
  against the framework's actual imports/registration.
* The Kaggle submission format for this competition has not been checked.
