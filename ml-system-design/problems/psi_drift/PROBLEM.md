# Detect drift with the population stability index

Write `solve(baseline, current, edges)` that measures how much a feature's distribution has shifted.

- `baseline`: the feature's values in the training data, a non-empty list of floats.
- `current`: the values seen in production this week, a non-empty list of floats.
- `edges`: sorted bin edges (a list of floats, possibly empty). With `m` edges there are `m + 1` bins; a value `x` goes into bin `bisect_right(edges, x)`, so a value equal to an edge goes to the bin on its right.

Compute each bin's share of `baseline`, `e_i`, and of `current`, `a_i`. Replace any share of exactly `0` by `0.0001` (do not renormalise). Then

```
psi = Σ (a_i − e_i) * ln(a_i / e_i)
```

Return `(psi, verdict)`: `psi` a float, and `verdict` the string `"stable"` if `psi < 0.1`, `"moderate"` if `psi < 0.25`, else `"major"`.

```
solve([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 5, 6, 7, 7, 8, 8], [2.5, 4.5, 6.5])
-> (2.1286, "major")
```

Baseline shares are `[0.25, 0.25, 0.25, 0.25]`. Current shares are `[0.25, 0, 0.25, 0.5]`; the empty bin becomes `0.0001` and contributes most of the index.

#### What interviewers look for

- Knowing what drift monitoring is for: labels arrive late, so input and prediction distributions are the early warning.
- Fixed bins from the baseline (often its deciles) so that the comparison is stable from week to week.
- The zero-share problem and a stated fix, and knowing PSI's usual thresholds are conventions, not statistics.

#### Follow-ups

- PSI is high for one feature but the model's online metric is flat. What do you do?
- How would you choose the bin edges, and what goes wrong with equal-width bins on a long-tailed feature?
- How would you monitor drift for an embedding or for text inputs?
