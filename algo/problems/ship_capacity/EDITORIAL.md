#### The idea
"Find the smallest X such that X works" is a binary search whenever *works* is monotone. Here a larger truck never
needs more days, so the feasible capacities are a suffix `[answer, ∞)`:

<figure class="diagram"><svg viewBox="0 0 560 110" width="560" role="img" aria-label="Capacities from max weight to total weight, infeasible then feasible">
<rect x="20" y="30" width="220" height="30" class="d-b"/><text x="130" y="50" class="d-text" text-anchor="middle">too small: more than D days</text>
<rect x="240" y="30" width="300" height="30" class="d-a"/><text x="390" y="50" class="d-text" text-anchor="middle">works: D days or fewer</text>
<text x="20" y="80" class="d-muted" text-anchor="start">max(weights)</text>
<text x="540" y="80" class="d-muted" text-anchor="end">sum(weights)</text>
<text x="240" y="98" class="d-text d-bold" text-anchor="middle">answer</text>
</svg>
<figcaption>Binary search finds the boundary with about log₂(sum) greedy checks.</figcaption></figure>

The check itself is greedy: fill each day until the next package would not fit. Greedy is optimal for a fixed
capacity, because taking a package today never makes a later day worse.

#### A solution
```python
def solve(weights, days):
    def days_needed(capacity):
        used, load = 1, 0
        for w in weights:
            if load + w > capacity:
                used += 1
                load = 0
            load += w
        return used

    lo, hi = max(weights), sum(weights)
    while lo < hi:                       # invariant: the answer is in [lo, hi]
        mid = (lo + hi) // 2
        if days_needed(mid) <= days:
            hi = mid                     # mid works: the answer is mid or smaller
        else:
            lo = mid + 1                 # mid fails: the answer is larger
    return lo
```

#### What it costs
`O(n · log(sum − max))`: about 30 greedy passes over 100,000 packages for the benchmark. Trying every capacity from
the heaviest package upward is `O(n · answer)`. The same shape solves "split an array into k parts minimising the
largest sum", "Koko eating bananas", the minimum speed or time for a deadline, and, in ML systems, the smallest batch
size or GPU count that meets a throughput target when you can only simulate a candidate.

#### Pitfalls
- Starting the search at 1 instead of `max(weights)`: the greedy check then loops forever or miscounts, because a
  package heavier than `C` never fits.
- `hi = mid - 1` when `mid` works, which can skip the answer.
- Off-by-one in the day count: start at 1 day, not 0.
- Using fewer days is allowed, so the test is `<= days`, not `== days`.
