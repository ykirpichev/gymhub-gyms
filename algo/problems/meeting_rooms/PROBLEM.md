# Meeting rooms

**Level:** medium   **Teaches:** sweep lines over events, sorting two arrays, min-heaps

## Task

Each meeting is a half-open interval `[start, end)`: it occupies a room from `start` up to, but not including, `end`. Return the smallest number of rooms that can hold every meeting, with no two meetings in the same room at the same time.

## Arguments

| Name       | Type              | Role                                                 |
| ---------- | ----------------- | ---------------------------------------------------- |
| `meetings` | `list[list[int]]` | pairs `[start, end]` with `start < end`, unsorted    |
| returns    | `int`             | the minimum number of rooms (0 for no meetings)      |

## Rules

- A meeting that ends at time `t` frees its room for a meeting that starts at `t`. `[1, 4)` and `[4, 6)` can share a room.
- Times may be negative and may repeat.

## Example

```
meetings = [[0, 30], [5, 10], [15, 20], [10, 15]]
at time 5 two meetings run: [0, 30) and [5, 10)
[5, 10) ends at 10, exactly when [10, 15) starts, so they share a room
result = 2
```

## How to think about it

The answer is the largest number of meetings running at the same moment. That number only changes at a start or an end, so look at those events in time order and keep a running count. At equal times, process ends before starts, because a room freed at `t` can be reused at `t`.

## Tests

Empty input, one meeting, back-to-back meetings, identical meetings, nested meetings, negative times, and random lists up to 3,000 meetings. Benchmark input: 200,000 random meetings. There is no time limit, but checking every pair of meetings is quadratic and very slow.
