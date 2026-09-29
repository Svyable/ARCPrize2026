"""Tiny stand-in environments with the same observable contract as ARC-AGI-3
(frame grid 0-15, NOT_FINISHED/WIN/GAME_OVER, levels-completed score, per-game
available actions). Used only for local testing; they say nothing about real games."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    frame: list  # [grid]
    state: str = "NOT_FINISHED"
    score: int = 0
    available_actions: list = field(default_factory=list)


class MockGame:
    """Base: subclasses implement _load(level), _apply(action, x, y), _render()."""

    n_levels = 3
    available = [1, 2, 3, 4, 5]
    hud_bar = False

    def __init__(self) -> None:
        self.level = 0
        self.steps_in_level = 0
        self.state = "NOT_PLAYED"
        self.optimal: list[int] = []  # optimal action count per level (for scoring)

    def reset(self) -> Frame:
        if self.state in ("NOT_PLAYED", "WIN"):
            self.level = 0
        self._load(self.level)
        self.steps_in_level = 0
        self.state = "NOT_FINISHED"
        return self._frame()

    def step(self, action_id: int, x: int | None = None, y: int | None = None) -> Frame:
        if action_id == 0:
            return self.reset()
        if self.state != "NOT_FINISHED":
            return self._frame()
        self.steps_in_level += 1
        outcome = self._apply(action_id, x, y)  # None | "win" | "dead"
        if outcome == "dead":
            self.state = "GAME_OVER"
        elif outcome == "win":
            self.level += 1
            if self.level >= self.n_levels:
                self.state = "WIN"
            else:
                self._load(self.level)
                self.steps_in_level = 0
        return self._frame()

    def _frame(self) -> Frame:
        g = self._render()
        if self.hud_bar:
            g[-1, :] = 0
            g[-1, : max(0, g.shape[1] - self.steps_in_level)] = 11
        return Frame([g.tolist()], self.state, self.level, list(self.available))


class MazeGame(MockGame):
    """Move a player (9) to the goal (3) through walls (5); red cells (2) kill.
    Actions 1-4 = up/down/left/right, 5 = no-op. Optional shrinking HUD bar."""

    LEVELS = [
        ["#########",
         "#P..#...#",
         "#.#.#.#.#",
         "#.#...#G#",
         "#########"],
        ["###########",
         "#P.....#..#",
         "#.####.#X##",
         "#.#..#...G#",
         "#.#.##.####",
         "#...X.....#",
         "###########"],
        ["#############",
         "#P#.....#...#",
         "#.#.###.#.#.#",
         "#.#.#X#...#.#",
         "#.#.#.#####.#",
         "#...#.....#G#",
         "#XXX#####.#.#",
         "#.........#.#",
         "#############"],
    ]

    def __init__(self, hud_bar: bool = False) -> None:
        super().__init__()
        self.hud_bar = hud_bar
        self.n_levels = len(self.LEVELS)
        self.optimal = [self._bfs(i) for i in range(self.n_levels)]

    def _parse(self, level: int):
        rows = self.LEVELS[level]
        walls, hazards, player, goal = set(), set(), None, None
        for y, r in enumerate(rows):
            for x, c in enumerate(r):
                if c == "#":
                    walls.add((x, y))
                elif c == "X":
                    hazards.add((x, y))
                elif c == "P":
                    player = (x, y)
                elif c == "G":
                    goal = (x, y)
        return walls, hazards, player, goal, (len(rows[0]), len(rows))

    def _bfs(self, level: int) -> int:
        walls, hazards, p, g, _ = self._parse(level)
        q, seen = deque([(p, 0)]), {p}
        while q:
            (x, y), d = q.popleft()
            if (x, y) == g:
                return d
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                n = (x + dx, y + dy)
                if n not in walls and n not in hazards and n not in seen:
                    seen.add(n)
                    q.append((n, d + 1))
        raise ValueError("unsolvable")

    def _load(self, level: int) -> None:
        self.walls, self.hazards, self.pos, self.goal, self.size = self._parse(level)

    def _apply(self, a, x, y):
        d = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(a)
        if d:
            n = (self.pos[0] + d[0], self.pos[1] + d[1])
            if n not in self.walls:
                self.pos = n
        if self.pos in self.hazards:
            return "dead"
        if self.pos == self.goal:
            return "win"
        return None

    def _render(self) -> np.ndarray:
        w, h = self.size
        g = np.zeros((h + (1 if self.hud_bar else 0), w), dtype=np.uint8)
        for x, y in self.walls:
            g[y, x] = 5
        for x, y in self.hazards:
            g[y, x] = 2
        g[self.goal[1], self.goal[0]] = 3
        g[self.pos[1], self.pos[0]] = 9
        return g


class ClickGame(MockGame):
    """Click every red (2) tile to turn it green (3); other clicks do nothing.
    Only ACTION6 is available. Tiles are small blobs on a black board."""

    available = [6]
    TILES = [
        [(3, 3), (10, 5), (6, 12)],
        [(2, 2), (12, 2), (2, 12), (12, 12)],
        [(7, 1), (1, 7), (13, 7), (7, 13), (7, 7)],
    ]

    def __init__(self) -> None:
        super().__init__()
        self.n_levels = len(self.TILES)
        self.optimal = [len(t) for t in self.TILES]

    def _load(self, level: int) -> None:
        self.tiles = {t: False for t in self.TILES[level]}

    def _apply(self, a, x, y):
        if a != 6 or x is None:
            return None
        for (tx, ty) in self.tiles:
            if tx <= x <= tx + 1 and ty <= y <= ty + 1:
                self.tiles[(tx, ty)] = True
        return "win" if all(self.tiles.values()) else None

    def _render(self) -> np.ndarray:
        g = np.zeros((16, 16), dtype=np.uint8)
        g[0, :] = 1  # static frame decoration
        for (tx, ty), done in self.tiles.items():
            g[ty : ty + 2, tx : tx + 2] = 3 if done else 2
        return g
