import random
from collections import deque

from gymkit.spec import Problem

TIME_LIMIT_MS = {"python": 800}


def reference(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    width = cols
    blocked = bytearray(rows * cols)
    for r, line in enumerate(grid):
        for c, ch in enumerate(line):
            if ch == "#":
                blocked[r * width + c] = 1
    source, target = start[0] * width + start[1], goal[0] * width + goal[1]
    dist = [-1] * (rows * cols)
    dist[source] = 0
    queue = deque([source])
    while queue:
        cell = queue.popleft()
        if cell == target:
            return dist[cell]
        r, c = divmod(cell, width)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols:
                nxt = nr * width + nc
                if not blocked[nxt] and dist[nxt] == -1:
                    dist[nxt] = dist[cell] + 1
                    queue.append(nxt)
    return -1


def _random_case(seed, rows, cols, wall, name):
    rng = random.Random(seed)
    cells = [["#" if rng.random() < wall else "." for _ in range(cols)] for _ in range(rows)]
    for c in range(cols):
        cells[0][c] = "."  # an open top row and right column: the goal is always reachable,
    for r in range(rows):
        cells[r][cols - 1] = "."  # but BFS still explores most of the grid before it gets there
    return {"_name": name, "grid": ["".join(row) for row in cells], "start": [0, 0], "goal": [rows - 1, cols - 1]}


def functional_cases():
    return [
        {"_name": "example", "grid": ["..#.", ".##.", "...."], "start": [0, 0], "goal": [0, 3]},
        {"_name": "start is goal", "grid": ["...", "..."], "start": [1, 1], "goal": [1, 1]},
        {"_name": "one cell", "grid": ["."], "start": [0, 0], "goal": [0, 0]},
        {"_name": "walled off", "grid": [".#.", "##.", "..."], "start": [0, 0], "goal": [2, 2]},
        {"_name": "single row", "grid": ["......"], "start": [0, 5], "goal": [0, 0]},
        {"_name": "long detour",
         "grid": [".#....", ".#.##.", ".#..#.", ".##.#.", "....#."], "start": [0, 0], "goal": [0, 5]},
        {"_name": "open grid", "grid": ["....." for _ in range(5)], "start": [0, 0], "goal": [4, 4]},
        _random_case(1, 12, 15, 0.3, "random 12x15"),
        _random_case(2, 80, 90, 0.25, "random 80x90"),
        _random_case(5, 60, 60, 0.45, "random 60x60, dense walls"),
    ]


def performance_case(seed: int = 0):
    return _random_case(seed, 600, 600, 0.25, "perf")


def build() -> Problem:
    return Problem(
        slug="grid_shortest_path",
        title="Shortest path in a grid",
        difficulty="medium",
        summary="Breadth-first search on a grid with walls: the fewest steps from start to goal.",
        params=["grid", "start", "goal"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        time_limit_ms=TIME_LIMIT_MS,
        tags=["graphs", "bfs", "grids"],
    )
