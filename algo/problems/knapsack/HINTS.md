# Hints for 0/1 knapsack

## Hint 1

Greedy by value per weight fails: with budget 10 and items of weight 6 (value 8), 5 (value 6) and 5 (value 6), greedy takes the weight-6 item first (the best ratio), then nothing else fits: value 8. The two weight-5 items give 12. Think in terms of budgets instead.

## Hint 2

`best[c]` for `c = 0 .. capacity`, all starting at 0. For each item `(w, v)`: `best[c] = max(best[c], best[c - w] + v)` for every `c >= w`.

## Hint 3

Run `c` from `capacity` down to `w`. Going upwards would let the same item be added again on top of itself (that is the unbounded knapsack).
