"""Frame analysis: connected components and click-target proposals."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Component:
    color: int
    size: int
    x: int  # column of the representative cell
    y: int  # row of the representative cell


def background_color(grid: np.ndarray) -> int:
    return int(np.bincount(grid.ravel(), minlength=16).argmax())


def components(grid: np.ndarray, skip_color: int | None = None) -> list[Component]:
    """4-connected same-colour components. The representative cell is the member
    closest to the centroid, so ring-shaped objects still yield a cell inside them."""
    h, w = grid.shape
    g = grid.tolist()
    seen = [[False] * w for _ in range(h)]
    out: list[Component] = []
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0][x0]:
                continue
            c = g[y0][x0]
            seen[y0][x0] = True
            if c == skip_color:
                continue
            cells = [(x0, y0)]
            stack = [(x0, y0)]
            while stack:
                x, y = stack.pop()
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and g[ny][nx] == c:
                        seen[ny][nx] = True
                        cells.append((nx, ny))
                        stack.append((nx, ny))
            cx = sum(p[0] for p in cells) / len(cells)
            cy = sum(p[1] for p in cells) / len(cells)
            rx, ry = min(cells, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
            out.append(Component(c, len(cells), rx, ry))
    return out


def click_candidates(
    grid: np.ndarray, max_components: int = 48, grid_points: int = 4
) -> list[tuple[int, int]]:
    """Ordered (x, y) click targets. Objects first (rare colours, then small size:
    interactive pieces tend to be small and uncommon), then a coarse lattice as a
    fallback for hidden hot-spots."""
    h, w = grid.shape
    bg = background_color(grid)
    comps = components(grid, skip_color=bg)
    color_cells = np.bincount(grid.ravel(), minlength=16)
    comps.sort(key=lambda c: (int(color_cells[c.color]), c.size, c.y, c.x))
    pts: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for c in comps[:max_components]:
        p = (c.x, c.y)
        if p not in seen:
            seen.add(p)
            pts.append(p)
    for i in range(grid_points):
        for j in range(grid_points):
            p = (int((j + 0.5) * w / grid_points), int((i + 0.5) * h / grid_points))
            if p not in seen:
                seen.add(p)
                pts.append(p)
    return pts
