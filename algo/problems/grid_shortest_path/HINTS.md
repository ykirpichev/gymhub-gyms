# Hints for Shortest path in a grid

## Hint 1

Treat each open cell as a node with up to four neighbours. All edges have the same length, so you do not need Dijkstra.

## Hint 2

Use `collections.deque` as a FIFO queue of cells, and a distance array (or dict) that doubles as the visited set.

## Hint 3

Set the distance of a neighbour as soon as you enqueue it. Return as soon as the goal is dequeued (or enqueued). If the queue empties first, return `-1`.
