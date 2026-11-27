# Dijkstra shortest paths

**Level:** medium   **Teaches:** adjacency lists, priority queues, lazy deletion

## Task

Given a directed graph with non-negative edge weights, find the length of the shortest path from `src` to every node. Nodes you cannot reach get `-1`.

## Arguments

| Name    | Type              | Role                                                    |
| ------- | ----------------- | ------------------------------------------------------- |
| `n`     | `int`             | number of nodes, labeled `0 .. n-1`                     |
| `edges` | `list[list[int]]` | `[u, v, w]`: an edge from `u` to `v` of weight `w >= 0` |
| `src`   | `int`             | start node                                              |
| returns | `list[int]`       | `dist[i]` for every node, `-1` if unreachable           |

Solve it in Python or C++. In C++ the function is:

```cpp
vector<long long> solve(int n, vector<vector<int>>& edges, int src);
```

Each edge is a 3-element `vector<int>` `{u, v, w}`. Path lengths can pass 2³¹, so keep distances in `long long`.

## Rules

- Edges are one-way. `[u, v, w]` does not let you go from `v` to `u`.
- Parallel edges and self loops can appear. Weights can be `0`.
- `dist[src] = 0`.

## Example

```
n = 5   src = 0
edges: 0->1 (4)  0->2 (1)  2->1 (2)  1->3 (1)  2->3 (5)
dist[1] = 3 via 0->2->1 (shorter than the direct 4)
dist[3] = 4 via 0->2->1->3
node 4 has no incoming edges
result = [0, 3, 1, 4, -1]
```

## How to think about it

Grow the set of settled nodes in order of distance. Each step settles the unsettled node with the smallest known distance; because weights are non-negative, no later path can beat it. A binary heap (`heapq`) of `(distance, node)` pairs finds that node in O(log n). Python's heap has no decrease-key, so push a new pair whenever you find a shorter distance and skip stale pairs when they pop. Total cost: O((n + m) log m). Scanning all nodes for the minimum each step is O(n²), which is hopeless at 100,000 nodes.

## Tests

A single node, no edges, direction matters, zero weights and self loops, parallel edges, a case where the cheapest first edge is a trap, and random graphs. Benchmark input: `n = 100,000` nodes and `m = 300,000` edges, weights up to 10⁶. `bench` fails if the median is over **1200 ms** in Python or **70 ms** in C++.
