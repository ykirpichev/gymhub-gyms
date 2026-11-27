# Minimum shipping capacity

**Level:** medium   **Teaches:** binary search on the answer, monotone feasibility checks, greedy packing

## Task

Packages must leave a warehouse **in the given order**. Each day a truck takes the next packages in line, as many as fit within its capacity `C` (the sum of their weights is at most `C`); a package is never split. Return the smallest integer capacity `C` that ships every package within `days` days.

## Arguments

| Name      | Type        | Role                                               |
| --------- | ----------- | -------------------------------------------------- |
| `weights` | `list[int]` | package weights in order, each `>= 1`; at least one |
| `days`    | `int`       | days available, `1 <= days <= len(weights)`        |
| returns   | `int`       | the smallest capacity that works                   |

## Rules

- The capacity must be at least the heaviest package, or that package can never ship.
- A day may end early: loading stops when the next package would exceed `C`.
- Using fewer than `days` days is fine.

## Example

```
weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   days = 5
C = 15:  [1..5] [6, 7] [8] [9] [10]      5 days   works
C = 14:  [1..4] [5, 6] [7] [8] [9] [10]  6 days   too many
result = 15
```

## How to think about it

You cannot compute `C` directly, but you can check one guess quickly: load greedily and count the days. And the check is monotone: if `C` works, every larger capacity works too. So binary-search the smallest `C` that works, between `max(weights)` and `sum(weights)`.

## Tests

One package, one day, as many days as packages, equal weights, one heavy package among light ones, and random inputs up to 2,000 packages. Benchmark input: 100,000 packages with weights up to 10,000 and `days = 500`. `bench` fails if the median is over **600 ms** in Python; trying every capacity from the heaviest package upward takes far longer.
