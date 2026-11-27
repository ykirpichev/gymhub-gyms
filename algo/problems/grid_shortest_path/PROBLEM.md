# Shortest path in a grid

**Level:** medium   **Teaches:** breadth-first search, grids as implicit graphs, visited sets

## Task

`grid` is a list of equal-length strings. `"."` is an open cell and `"#"` is a wall. From `start`, move one cell up, down, left or right per step, never into a wall or off the grid. Return the fewest steps to reach `goal`, or `-1` if it cannot be reached.

## Arguments

| Name    | Type        | Role                                                   |
| ------- | ----------- | ------------------------------------------------------ |
| `grid`  | `list[str]` | `R` rows of `C` characters, each `.` or `#`; `R, C >= 1` |
| `start` | `list[int]` | `[row, col]` of an open cell                           |
| `goal`  | `list[int]` | `[row, col]` of an open cell                           |
| returns | `int`       | the number of steps, `0` if `start == goal`, `-1` if unreachable |

## Example

```
grid = ["..#.",
        ".##.",
        "...."]
start = [0, 0]   goal = [0, 3]
path: (0,0) (1,0) (2,0) (2,1) (2,2) (2,3) (1,3) (0,3)
result = 7
```

## How to think about it

Every step costs the same, so breadth-first search finds shortest paths: it visits all cells at distance 1, then all at distance 2, and so on. Mark a cell visited when you **add** it to the queue, not when you take it out, or the same cell is queued many times.

## Tests

Start equals goal, a one-cell grid, a goal behind a wall, a single row, a maze with a long detour, an open grid, and random grids. Benchmark input: a 600 × 600 grid with about 25% walls. `bench` fails if the median is over **800 ms** in Python.
