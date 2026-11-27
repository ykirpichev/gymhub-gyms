# Merge intervals

**Level:** easy   **Teaches:** sort-then-sweep, interval overlap tests

## Task

You get a list of closed intervals `[start, end]` in no particular order. Combine every group of intervals that overlap or touch into one interval, and return the result sorted by start. Two intervals touch when one ends at the exact point where the other starts, as in `[1, 4]` and `[4, 5]`.

## Arguments

| Name        | Type              | Role                                             |
| ----------- | ----------------- | ------------------------------------------------ |
| `intervals` | `list[list[int]]` | pairs `[start, end]` with `start <= end`, unsorted |
| returns     | `list[list[int]]` | disjoint intervals, sorted by start              |

## Rules

- Intervals are inclusive on both ends.
- `[a, b]` and `[c, d]` with `a <= c` belong together when `c <= b`.
- The output must not contain two intervals that overlap or touch.
- An empty input gives an empty output.

## Example

```
intervals = [[8, 10], [1, 3], [2, 6], [15, 18]]
sorted    = [[1, 3], [2, 6], [8, 10], [15, 18]]
[1, 3] and [2, 6] overlap  ->  [1, 6]
result    = [[1, 6], [8, 10], [15, 18]]
```

## How to think about it

Once the intervals are sorted by start, any interval can only merge with the one you built just before it. Keep the last merged interval open and either stretch its end or start a new one. Sorting dominates the cost: O(n log n) time.

## Tests

Empty input, a single interval, touching ends, nested intervals, duplicates, negative coordinates, and random lists up to 2,000 intervals. Benchmark input: 200,000 random intervals. There is no time limit, but a quadratic pairwise merge will be very slow.
