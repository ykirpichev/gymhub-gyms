#### The idea
Keep the first `k` items. For item `i` (0-based) after that, keep it with probability `k / (i + 1)` by drawing
`j = randrange(i + 1)` and replacing slot `j` when `j < k`.

Why it is uniform: suppose after `i` items every item is in the reservoir with probability `k / i`. Item `i` enters
with probability `k / (i + 1)`. An item already in the reservoir survives step `i` unless item `i` enters **and** lands
on its slot: `1 − (k/(i+1))·(1/k) = i/(i+1)`. So it is still there with probability `(k/i)·(i/(i+1)) = k/(i+1)`. The
induction holds.

#### A solution
```python
import random


def solve(stream, k, seed):
    if k < 0:
        raise ValueError("k must be non-negative")
    rng = random.Random(seed)
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            # Keep item i with probability k / (i + 1); it replaces a uniformly chosen slot.
            j = rng.randrange(i + 1)
            if j < k:
                reservoir[j] = item
    return reservoir
```

#### What it costs
`O(n)` time, one random draw per item past the first `k`, and `O(k)` memory. "Algorithm L" skips ahead by a
geometrically distributed number of items, which needs only `O(k·(1 + log(n/k)))` random draws. For a weighted sample,
give each item the key `u^(1/w)` with `u` uniform and keep the `k` largest keys in a heap (Efraimidis–Spirakis).

#### Pitfalls
- `list(stream)` first: correct and `O(n)` memory, which is exactly what reservoir sampling exists to avoid.
- `randrange(i)` instead of `randrange(i + 1)`: off by one, and the bias shows up in the frequency test.
- Appending instead of replacing, so the list grows past `k`.
- Creating the generator from the global `random`, so the same seed does not reproduce the sample.
