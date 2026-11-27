# Minimum spanning tree

**Level:** medium   **Teaches:** Kruskal's algorithm, greedy choice with a cut argument, union-find in practice

## Task

An undirected graph has `n` nodes `0 .. n-1` and weighted edges. A spanning tree connects all nodes with `n - 1` edges and no cycle. Return the smallest possible total weight of a spanning tree, or `-1` if the graph is not connected.

## Arguments

| Name    | Type              | Role                                                      |
| ------- | ----------------- | --------------------------------------------------------- |
| `n`     | `int`             | number of nodes, `>= 1`                                   |
| `edges` | `list[list[int]]` | `[u, v, w]`: an undirected edge of weight `w` (may be negative or 0) |
| returns | `int`             | the minimum total weight, `0` when `n == 1`, `-1` when disconnected |

## Rules

- Parallel edges and self-loops can appear. A self-loop never belongs to a spanning tree.
- Weights can be negative; the tree still has exactly `n - 1` edges.

## Example

```
n = 4   edges = [[0, 1, 1], [1, 2, 4], [0, 2, 3], [2, 3, 2], [1, 3, 5]]
sorted by weight: (0,1,1) (2,3,2) (0,2,3) (1,2,4) (1,3,5)
take 0-1 (1), take 2-3 (2), take 0-2 (3): all four nodes joined
1-2 and 1-3 would close a cycle
result = 6
```

## How to think about it

Take edges from lightest to heaviest, and keep one whenever it joins two different components. That greedy choice is always safe: the lightest edge across any split of the nodes belongs to some minimum spanning tree. Union-find answers "different components?" in near-constant time. Stop once you have `n - 1` edges.

## Tests

One node, two nodes, a disconnected graph, parallel edges, self-loops, negative weights, ties, and random connected graphs. Benchmark input: `n = 100,000` with 300,000 edges. `bench` fails if the median is over **1500 ms** in Python.
