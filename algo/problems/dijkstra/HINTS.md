# Hints for Dijkstra shortest paths

## Hint 1

First turn the edge list into an adjacency list: `graph = [[] for _ in range(n)]`, then `graph[u].append((v, w))` for every edge. Scanning the whole edge list per node is far too slow.

## Hint 2

Keep a heap of `(dist, node)` starting with `(0, src)`. Pop the smallest. If that node is already settled, skip it; otherwise settle it and push `(dist + w, v)` for each outgoing edge that improves `v`.

## Hint 3

Use a single `dist` list for both "best known" and "settled" checks, and compare against it before pushing to keep the heap small. Bind `heapq.heappush` and `heapq.heappop` to local names inside the function; attribute lookups in a hot loop add up in Python. Convert the never-reached `inf` values to `-1` at the end.

In C++, use `priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>>` for a min-heap. Building the adjacency lists once as flat arrays (offsets per node, then targets and weights) is faster than `vector<vector<...>>`.
