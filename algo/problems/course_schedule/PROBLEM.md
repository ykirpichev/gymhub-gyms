# Course schedule

**Level:** medium   **Teaches:** topological sort (Kahn's algorithm), cycle detection, heaps for a canonical order

## Task

There are `n` courses labelled `0 .. n-1`. Each prerequisite `[a, b]` says course `b` must be taken before course `a`. Return an order that takes every course after all of its prerequisites. Many orders may exist; return the **lexicographically smallest** one: at every point, take the smallest-numbered course whose prerequisites are all done. If no order exists (the prerequisites form a cycle), return `[]`.

## Arguments

| Name            | Type              | Role                                          |
| --------------- | ----------------- | --------------------------------------------- |
| `n`             | `int`             | number of courses, `>= 1`                     |
| `prerequisites` | `list[list[int]]` | pairs `[a, b]`: take `b` before `a`           |
| returns         | `list[int]`       | all `n` courses in order, or `[]` on a cycle  |

## Rules

- A pair may repeat. A course listed as its own prerequisite (`[a, a]`) is a cycle.
- Courses with no prerequisites can be taken at any time.

## Example

```
n = 4   prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]
0 has none: take 0. Now 1 and 2 are free: take 1 (smaller), then 2. Then 3.
result = [0, 1, 2, 3]

n = 2   prerequisites = [[0, 1], [1, 0]]  ->  []
```

## How to think about it

Count each course's unfinished prerequisites (its in-degree). Courses at zero are ready. Take a ready course, and lower the count of every course that depended on it; any that reach zero become ready. A min-heap of ready courses gives the smallest-first rule. If you run out of ready courses before taking all `n`, the rest sit on a cycle.

## Tests

One course, no prerequisites, a chain, a diamond, repeated pairs, a self-loop, a cycle among some courses, and random graphs. Benchmark input: `n = 100,000` with 300,000 prerequisites and no cycle. `bench` fails if the median is over **800 ms** in Python.
