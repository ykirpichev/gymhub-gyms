#### The idea
A grid is a graph whose edges you never build: a cell's neighbours are computed on the fly. With every step costing 1,
BFS explores cells in **layers** of equal distance, so the first time it reaches the goal is along a shortest path.

<figure class="diagram"><svg viewBox="0 0 520 150" width="520" role="img" aria-label="BFS layers spreading from the start around a wall">
<rect x="20" y="20" width="34" height="34" class="d-c"/><text x="37" y="42" class="d-text" text-anchor="middle">0</text>
<rect x="56" y="20" width="34" height="34" class="d-a"/><text x="73" y="42" class="d-text" text-anchor="middle">1</text>
<rect x="92" y="20" width="34" height="34" class="d-fill"/><text x="109" y="42" class="d-text" text-anchor="middle">#</text>
<rect x="128" y="20" width="34" height="34" class="d-a"/><text x="145" y="42" class="d-text" text-anchor="middle">7</text>
<rect x="20" y="56" width="34" height="34" class="d-a"/><text x="37" y="78" class="d-text" text-anchor="middle">1</text>
<rect x="56" y="56" width="34" height="34" class="d-fill"/><text x="73" y="78" class="d-text" text-anchor="middle">#</text>
<rect x="92" y="56" width="34" height="34" class="d-fill"/><text x="109" y="78" class="d-text" text-anchor="middle">#</text>
<rect x="128" y="56" width="34" height="34" class="d-a"/><text x="145" y="78" class="d-text" text-anchor="middle">6</text>
<rect x="20" y="92" width="34" height="34" class="d-a"/><text x="37" y="114" class="d-text" text-anchor="middle">2</text>
<rect x="56" y="92" width="34" height="34" class="d-a"/><text x="73" y="114" class="d-text" text-anchor="middle">3</text>
<rect x="92" y="92" width="34" height="34" class="d-a"/><text x="109" y="114" class="d-text" text-anchor="middle">4</text>
<rect x="128" y="92" width="34" height="34" class="d-a"/><text x="145" y="114" class="d-text" text-anchor="middle">5</text>
<text x="190" y="50" class="d-text" text-anchor="start">Each number is the BFS distance from the start (0).</text>
<text x="190" y="72" class="d-text" text-anchor="start">The queue holds one layer, then the next.</text>
<text x="190" y="94" class="d-muted" text-anchor="start">The goal at the top right is reached in 7 steps.</text>
</svg>
<figcaption>The example grid with the distance BFS assigns to every open cell.</figcaption></figure>

#### A solution
```python
from collections import deque


def solve(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    dist = [[-1] * cols for _ in range(rows)]
    sr, sc = start
    dist[sr][sc] = 0
    queue = deque([(sr, sc)])
    while queue:
        r, c = queue.popleft()
        if [r, c] == goal:
            return dist[r][c]
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "." and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1      # mark on enqueue
                queue.append((nr, nc))
    return -1
```

#### What it costs
`O(R·C)` time and memory: each cell is enqueued once and checks four neighbours. Weighted cells need Dijkstra; 0/1 weights
(free moves and paid moves) need a deque where free moves go to the front. A* with the Manhattan distance as the
heuristic explores fewer cells when the goal is far from the start. Bidirectional BFS from both ends roughly halves the
explored area again.

#### Pitfalls
- Marking visited on dequeue: a cell can be queued by several neighbours, and the queue grows far larger.
- Using a list with `pop(0)`, which is `O(n)` per pop.
- Recursive DFS: it finds **a** path, not the shortest, and hits the recursion limit on large grids.
- Swapping rows and columns when reading `grid[r][c]`.
