#### The idea
The state is the budget. `best[c]` holds the most value achievable with weight at most `c` using the items processed
so far. Item `(w, v)` gives every budget `c ≥ w` one new option, "take it", worth `best[c − w] + v` where
`best[c − w]` must not already include this item.

Iterating `c` **downwards** guarantees that: when `best[c]` is updated, `best[c − w]` (a smaller index) has not been
touched in this item's pass yet.

| budget c | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| after item (1, 1) | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| after item (3, 4) | 0 | 1 | 1 | 4 | 5 | 5 | 5 | 5 |
| after item (4, 5) | 0 | 1 | 1 | 4 | 5 | 6 | 6 | 9 |
| after item (5, 7) | 0 | 1 | 1 | 4 | 5 | 7 | 8 | 9 |

#### A solution
```python
def solve(weights, values, capacity):
    best = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):   # downwards: each item at most once
            take = best[c - w] + v
            if take > best[c]:
                best[c] = take
    return best[capacity]
```

#### What it costs
`O(n · capacity)` time and `O(capacity)` memory: 2 million updates for the benchmark. That is **pseudo-polynomial**:
polynomial in the numeric value of the budget, not in its number of digits, which is why knapsack is NP-hard yet easy
for small budgets. With NumPy, each item's pass is one vectorised `np.maximum(best[w:], best[:-w] + v)` on a copy,
about 50 times faster. When values are small and weights huge, flip the DP: the least weight for each total value.

#### Pitfalls
- Looping `c` upwards: that is the unbounded knapsack, and the "each item once" test catches it.
- Greedy by value per weight, which fails on the ratio test.
- A 2-D table `(n + 1) × (capacity + 1)` is correct but uses 200 times the memory; keep it only if you must
  reconstruct the chosen items.
- `range(capacity, w, -1)`, which skips budget `c = w`.
