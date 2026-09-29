# ARC Prize 2026 – ARC-AGI-3 agent

A graph-exploration agent for ARC-AGI-3 interactive games, its offline evaluation harness, and a
standalone competition runner. Pure Python + numpy: no LLM, no network at run time.

```
arcagent/core.py        the agent (framework-independent; duck-typed frames)
arcagent/vision.py      connected components -> ACTION6 click targets
arcagent/gym.py         Move -> arcengine GameAction
arcagent/submit.py      standalone runner: open scorecard, play every game, close scorecard
arcagent/mock_env.py    tiny mock games (mazes, HUD bar, step budget, clicks) for unit tests
ARC-AGI-3-Agents/       vendored framework, with agents/templates/explorer.py registered as `explorer`
evaluate.py             play the public games locally, score with the toolkit's formula
run_framework_offline.py  run the agent through the framework's own Swarm, no network
tools/                  dev-only oracles / profilers (see tools/README.md)
tests/                  pytest suite (17 tests)
```

## How it plays

Each level is treated as a deterministic state graph, rebuilt from scratch at every level-up.

* **Nodes = frames**, with volatile HUD cells masked out. A cell/row/column is volatile if it changes
  under *every* action id tried (bumps and no-ops included) — that is what a step bar or timer does,
  and what gameplay never does.
* **Edges = (state, action) results.** Every pair is tried at most once. ACTION6 targets are the
  centres of connected components (rare colours and small objects first) plus a coarse lattice.
* **Deaths.** GAME_OVER blocks the fatal edge and the agent RESETs; the graph survives, so later paths
  avoid it. But real games kill at a fixed *life length* (a step/energy budget of 50–200 actions), which
  the frame-only node key cannot see. Deaths repeating at one length on ≥3 different edges are taken as
  a budget and stop counting against their edge; when everything is exhausted, blocked edges get one
  retry (longest life first). With a known budget the planner RESETs early rather than walking to a
  frontier it cannot reach, but only if a fresh life could do better (no thrash loops).
* **Portfolio search.** For the first 250 actions of a level: greedy nearest-frontier with *momentum*
  (repeat an action that keeps changing the frame; action kinds ranked by how often they reach a
  never-seen state). Afterwards: breadth-first, where depth counts *runs* of one action (repeating
  costs 0.25), because optimal solutions in these games are made of runs (`3×3, 1×4, 4×3, 1×3`).
  The planner may use a virtual RESET edge (any state → level start).
* **Transfer.** Action-kind statistics carry over between levels; the graph does not.
* **Safety.** Any internal exception falls back to a random legal action; a per-game time budget and
  action cap end runs cleanly.

## Measured results (public games, this repo)

Local scores use the toolkit's own formula and match its scorecard exactly (verified).

| agent | actions/game | mean score | levels solved (of 183) |
|---|---|---|---|
| random | 2000 | 0.22 | – |
| **explorer** | 2000 | **0.29** | 13 |
| **explorer** | 6000 | 0.30 | 19 |

Read these honestly: **the score is close to zero.** Per-level score is `(human_baseline / actions)²`,
so a level solved in 10× the human count earns ~1%. Points come only from the handful of levels solved in
about the baseline count (lp85 7 vs 17, tn36 17 vs 32, vc33 19 vs 7). More actions solve more levels
(19 vs 13) but add almost no score. Most games (~16/25) are not solved at all.

What the oracle tools showed: winning sequences are *short* (3–18 actions; e.g. vc33 = click one object
3×, sp80 = `4,4,4,5`), but human baselines are 2–10× longer, and a blind search needs hundreds to
thousands of probes to find them. Closing that gap needs perception/rule inference (goal detection,
object roles, matching target patterns), not more search.

## Run

```bash
# Python >= 3.12 for the real games (arc-agi / arcengine); numpy + pytest for the rest
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install arc-agi arcengine numpy pytest

python -m pytest                                   # 17 tests (mock games + real-game integration)
python evaluate.py --max-actions 2000 --workers 8  # all 25 public games, toolkit-exact score
python run_local.py                                # mock games only

# standalone competition-style run (env vars follow the toolkit: OPERATION_MODE, ARC_API_KEY, ENVIRONMENTS_DIR)
OPERATION_MODE=offline ENVIRONMENTS_DIR=environment_files \
  python -m arcagent.submit --max-actions 5000 --time-budget 3600

# through the real ARC-AGI-3-Agents framework (needs its deps; pillow<=11.3)
python run_framework_offline.py ls20,tn36,vc33 500
```

## Submission notes / not verified

* **Kaggle format not verified.** Nothing here confirms how the competition harness invokes an agent
  (the task text mentions `arc_agi_3_wheels/`, which were not in the repo). `arcagent/submit.py` depends only on
  `arc_agi`, `arcengine` and numpy and follows the toolkit's `Arcade` API, so it should adapt to a notebook
  cell with little change — but it has only been run in offline mode against the 25 public games.
* Per-game action limits and wall-clock limits of the private evaluation are unknown; `--max-actions`
  and `--time-budget` are the knobs.
* The agent assumes determinism within a level. Games with randomness will merely make it re-probe.
* Tuning was done on the 25 public games only. Differences between variants there are a few levels
  and noisy; treat small score deltas as noise.
