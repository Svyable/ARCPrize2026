"""Framework-independent exploration agent for ARC-AGI-3.

Strategy (per level): treat the environment as a deterministic state graph.
  * Nodes are frames (with volatile HUD cells such as step bars masked out).
  * Every (node, action) pair is tried at most once; results become edges.
  * At a node with untried actions, take the one whose *kind* (simple action id, or
    click-on-colour) has most often changed the frame so far. Otherwise walk the
    shortest known path to the nearest node that still has untried actions.
  * GAME_OVER marks the last edge as fatal, then RESET; the graph survives the
    reset, so later paths route around known death edges.
  * A score increase (level completed) starts a fresh graph; learned action-kind
    statistics carry over because mechanics tend to persist across levels.
The only thing read from the environment is: grid, state name, score, available actions.
"""
from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .vision import click_candidates

log = logging.getLogger(__name__)

RESET_ID = 0
CLICK_ID = 6
DEAD = -1

Action = tuple[int, int, int]  # (action_id, x, y); x = y = -1 for non-click actions
_RESET: Action = (RESET_ID, -1, -1)


@dataclass(frozen=True)
class Move:
    action_id: int  # 0 = RESET, 1..7 = ACTIONn
    x: int | None = None
    y: int | None = None
    reason: str = ""


def _state_name(state: Any) -> str:
    return str(getattr(state, "name", state)).upper()


def read_frame(frame: Any) -> tuple[np.ndarray, str, int, list[int]]:
    """Duck-typed reader: works for the framework's FrameData, dicts and mocks."""
    get = (lambda k, d=None: frame.get(k, d)) if isinstance(frame, dict) else (lambda k, d=None: getattr(frame, k, d))
    raw = get("frame")
    arr = np.asarray(raw, dtype=np.uint8)
    if arr.ndim == 3:  # list of layers/animation frames -> final one
        arr = arr[-1]
    avail = [int(getattr(a, "value", a)) for a in (get("available_actions") or [])]
    score = int(get("score", get("levels_completed", 0)) or 0)
    return arr, _state_name(get("state", "NOT_FINISHED")), score, avail


class ExplorerAgent:
    def __init__(
        self,
        seed: int = 0,
        max_actions: int = 20000,
        max_components: int = 256,
        grid_points: int = 4,
        volatile_min_transitions: int = 6,
        volatile_threshold: float = 0.85,
        min_rate: float = 0.2,
    ) -> None:
        self.rng = random.Random(seed)
        self.max_actions = max_actions
        self.max_components = max_components
        self.grid_points = grid_points
        self.vmin = volatile_min_transitions
        self.vthr = volatile_threshold
        self.min_rate = min_rate
        self.actions_taken = 0
        self.score = 0
        self.won = False
        self.errors = 0
        self.stats: dict[tuple, list[int]] = {}  # kind -> [changed, tried]
        self._reset_level()

    # ---- framework-facing API -------------------------------------------------
    def is_done(self, frames: Any, latest_frame: Any) -> bool:
        if self.won or self.actions_taken >= self.max_actions:
            return True
        state = latest_frame.get("state") if isinstance(latest_frame, dict) else getattr(latest_frame, "state", None)
        return _state_name(state) == "WIN"

    def choose_action(self, frames: Any, latest_frame: Any) -> Move:
        try:
            grid, state, score, avail = read_frame(latest_frame)
            return self.step(grid, state, score, avail)
        except Exception:  # never let a bug abort a scored run: degrade to random play
            log.exception("explorer failed; falling back to a random action")
            self.errors += 1
            self._reset_level()  # graph may be corrupt; rebuild from scratch
            return self._random_move()

    def _random_move(self) -> Move:
        self.actions_taken += 1
        acts = [a for a in self.avail if a != RESET_ID] or [1]
        a = self.rng.choice(acts)
        if a == CLICK_ID:
            return Move(a, self.rng.randrange(64), self.rng.randrange(64), "fallback")
        return Move(a, reason="fallback")

    # ---- core step ------------------------------------------------------------
    def step(self, grid: np.ndarray, state: str, score: int, available: list[int]) -> Move:
        if state == "WIN":
            self.won = True
            return Move(RESET_ID, reason="won")
        if score > self.score:
            self.score = score
            self._reset_level()
        if score < self.score:  # framework restarted the whole game
            self.score = score
            self._reset_level()

        self.avail = available or self.avail
        if state in ("NOT_PLAYED", "NOT_STARTED") or not self.raws and state == "GAME_OVER":
            return self._emit(_RESET, "start")

        if state == "GAME_OVER":
            self._observe_death()
            return self._emit(_RESET, "game over")

        self._observe(grid)
        act = self._select()
        return self._emit(act, "explore")

    # ---- per-level state ------------------------------------------------------
    def _reset_level(self) -> None:
        self.raws: list[np.ndarray] = []
        self.raw_id: dict[bytes, int] = {}
        self.log: list[tuple[int, Action, int]] = []  # (src_raw, action, dst_raw | DEAD)
        self.nodes: dict[bytes, int] = {}
        self.node_raw: list[int] = []
        self.edges: list[dict[Action, int]] = []
        self.untried: list[list[Action]] = []
        self.cur_raw = -1
        self.cur_node = -1
        self.start_node = -1
        self.pending: tuple[int, Action] | None = None  # (src_raw, action)
        self.plan: deque[tuple[Action, int]] = deque()
        self.plan_expect: int | None = None
        self.mask: np.ndarray | None = None
        self.prev_grid: np.ndarray | None = None
        self._n_trans = 0
        self._vol: dict[int, dict] = {}
        self._vol_shape: tuple | None = None
        self.avail: list[int] = [1, 2, 3, 4, 5]

    def _emit(self, act: Action, reason: str) -> Move:
        self.actions_taken += 1
        aid, x, y = act
        if aid == CLICK_ID:
            return Move(aid, x, y, reason)
        return Move(aid, reason=reason)

    # ---- graph maintenance ----------------------------------------------------
    def _raw(self, grid: np.ndarray) -> int:
        b = grid.tobytes() + bytes(grid.shape)
        rid = self.raw_id.get(b)
        if rid is None:
            rid = len(self.raws)
            self.raw_id[b] = rid
            self.raws.append(grid.copy())
        return rid

    def _node(self, rid: int) -> int:
        g = self._masked(self.raws[rid])
        key = g.tobytes() + bytes(g.shape)
        n = self.nodes.get(key)
        if n is None:
            n = len(self.edges)
            self.nodes[key] = n
            self.node_raw.append(rid)
            self.edges.append({})
            self.untried.append(self._candidates(self.raws[rid]))
        return n

    def _candidates(self, grid: np.ndarray) -> list[Action]:
        acts: list[Action] = [(a, -1, -1) for a in self.avail if a not in (RESET_ID, CLICK_ID)]
        if CLICK_ID in self.avail:
            for x, y in click_candidates(grid, self.max_components, self.grid_points):
                acts.append((CLICK_ID, x, y))
        return acts

    def _kind(self, act: Action, grid: np.ndarray) -> tuple:
        if act[0] == CLICK_ID:
            return (CLICK_ID, int(grid[act[2], act[1]]))
        return (act[0],)

    def _rate(self, act: Action, grid: np.ndarray) -> tuple[bool, float]:
        """Sort key: simple actions tried < 2 times come first, then observed change rate."""
        ch, tr = self.stats.get(self._kind(act, grid), (0, 0))
        return (act[0] != CLICK_ID and tr < 2, (ch + 1) / (tr + 2))

    def _masked(self, g: np.ndarray) -> np.ndarray:
        if self.mask is not None and self.mask.shape == g.shape:
            return np.where(self.mask, 255, g).astype(np.uint8)
        return g

    def _observe(self, grid: np.ndarray) -> None:
        rid = self._raw(grid)
        if self.pending is not None:
            src_raw, act = self.pending
            self.pending = None
            self.log.append((src_raw, act, rid))
            changed = not np.array_equal(self._masked(self.raws[src_raw]), self._masked(grid))
            st = self.stats.setdefault(self._kind(act, self.raws[src_raw]), [0, 0])
            st[0] += int(changed)
            st[1] += 1
            self.cur_raw = rid
            rebuilt = self._update_volatility(self.raws[src_raw], grid, act)
            if rebuilt:
                self._rebuild()
            else:
                src_node = self.cur_node
                self.cur_node = self._node(rid)
                self._add_edge(src_node, act, self.cur_node)
        else:
            self.cur_raw = rid
            self.cur_node = self._node(rid)
        if self.start_node < 0:
            self.start_node = self.cur_node
        if self.plan_expect is not None and self.cur_node != self.plan_expect:
            self.plan.clear()
        self.plan_expect = None

    def _observe_death(self) -> None:
        if self.pending is None or not self.raws:
            return
        src_raw, act = self.pending
        self.pending = None
        self.log.append((src_raw, act, DEAD))
        self._add_edge(self.cur_node, act, DEAD)
        self.plan.clear()
        self.plan_expect = None
        # after RESET the frame is the level start; re-anchor on the next observation
        self.cur_node = -1

    def _add_edge(self, src: int, act: Action, dst: int) -> None:
        self.edges[src][act] = dst
        try:
            self.untried[src].remove(act)
        except ValueError:
            pass

    def _rebuild(self) -> None:
        self.nodes, self.node_raw, self.edges, self.untried = {}, [], [], []
        node_of_raw = [self._node(i) for i in range(len(self.raws))]
        for src, act, dst in self.log:
            self._add_edge(node_of_raw[src], act, DEAD if dst == DEAD else node_of_raw[dst])
        self.cur_node = node_of_raw[self.cur_raw]
        self.start_node = node_of_raw[0]
        self.plan.clear()

    # ---- volatile HUD detection ----------------------------------------------
    # A HUD element (step bar, timer, blinking cell) changes no matter which action is
    # taken -- including bumps into walls and no-ops -- whereas gameplay changes are
    # action-specific. So a border row/column, or a single cell, is volatile only if it
    # changed in >= vthr of the transitions of *every* action id tried >= 3 times.
    def _update_volatility(self, prev: np.ndarray, cur: np.ndarray, act: Action) -> bool:
        if prev.shape != cur.shape:
            return False
        if self._vol_shape != cur.shape:
            self._vol_shape = cur.shape
            self._vol = {}
        diff = prev != cur
        rec = self._vol.setdefault(act[0], {
            "n": 0,
            "cells": np.zeros(cur.shape, dtype=np.int32),
            "rows": np.zeros(cur.shape[0], dtype=np.int32),
            "cols": np.zeros(cur.shape[1], dtype=np.int32),
        })
        rec["n"] += 1
        rec["cells"] += diff
        rec["rows"] += diff.any(axis=1)
        rec["cols"] += diff.any(axis=0)
        self._n_trans += 1
        if self._n_trans < self.vmin or self._n_trans % 2:
            return False
        recs = [r for r in self._vol.values() if r["n"] >= 3]
        n_ids = len([a for a in self.avail if a != RESET_ID])
        if len(recs) < min(2, n_ids):
            return False
        h, w = cur.shape
        band_rows = [r for r in range(h) if r < 3 or r >= h - 3]
        band_cols = [c for c in range(w) if c < 3 or c >= w - 3]

        def ok(key: str) -> np.ndarray:
            out = np.ones(recs[0][key].shape, dtype=bool)
            for r in recs:
                out &= (r[key] / r["n"]) >= self.vthr
            return out

        new = ok("cells")
        rows_ok, cols_ok = ok("rows"), ok("cols")
        for r in band_rows:
            if rows_ok[r]:
                new[r, :] = True
        for c in band_cols:
            if cols_ok[c]:
                new[:, c] = True
        if self.mask is None or self.mask.shape != cur.shape:
            self.mask = np.zeros(cur.shape, dtype=bool)
        grew = bool((new & ~self.mask).any())
        if grew:
            self.mask |= new
        return grew

    # ---- action selection -----------------------------------------------------
    def _select(self) -> Action:
        node = self.cur_node
        best = self._best_untried(node) if self.untried[node] else None
        if best is not None and self._promising(best, node):
            return self._commit(node, best)
        if not self.plan:
            self.plan = self._bfs(node, promising_only=True)
        if not self.plan and best is not None:
            return self._commit(node, best)  # nothing promising anywhere: probe here
        if not self.plan:
            self.plan = self._bfs(node, promising_only=False)
        if self.plan:
            act, expect = self.plan.popleft()
            self.plan_expect = expect
            return self._commit(node, act)
        # Fully explored (or the rest is unreachable): restart the level, else poke randomly.
        if node != self.start_node:
            self.pending = None
            self.cur_node = -1
            return _RESET
        known = list(self.edges[node]) or self._candidates(self.raws[self.cur_raw])
        return self._commit(node, self.rng.choice(known))

    def _promising(self, act: Action, node: int) -> bool:
        """Worth trying now: an under-sampled simple action or a kind that often changes the frame."""
        forced, rate = self._rate(act, self.raws[self.node_raw[node]])
        return forced or rate >= self.min_rate

    def _node_promising(self, node: int) -> bool:
        return any(self._promising(a, node) for a in self.untried[node])

    def _commit(self, node: int, act: Action) -> Action:
        self.pending = (self.cur_raw, act)
        if act in self.untried[node]:
            self.untried[node].remove(act)
        return act

    def _best_untried(self, node: int) -> Action:
        lst = self.untried[node]
        g = self.raws[self.node_raw[node]]
        return max(enumerate(lst), key=lambda t: (self._rate(t[1], g), -t[0]))[1]

    def _bfs(self, start: int, promising_only: bool) -> deque[tuple[Action, int]]:
        parent: dict[int, tuple[int, Action]] = {}
        seen = {start}
        q = deque([start])
        while q:
            n = q.popleft()
            if n != start and self.untried[n] and (not promising_only or self._node_promising(n)):
                path: list[tuple[Action, int]] = []
                while n != start:
                    p, a = parent[n]
                    path.append((a, n))
                    n = p
                return deque(reversed(path))
            for a, d in self.edges[n].items():
                if d >= 0 and d != n and d not in seen:
                    seen.add(d)
                    parent[d] = (n, a)
                    q.append(d)
        return deque()
