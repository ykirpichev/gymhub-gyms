# Hints for Course schedule

## Hint 1

Build, for each course `b`, the list of courses that need it. Count how many prerequisites each course still waits for.

## Hint 2

Start with every course whose count is 0. Each time you take one, decrement the counts of the courses that need it.

## Hint 3

Use `heapq` for the ready courses so you always take the smallest. At the end, if fewer than `n` courses were taken, return `[]`. Count repeated pairs every time they appear, both in the adjacency list and in the in-degree, so they cancel out.
