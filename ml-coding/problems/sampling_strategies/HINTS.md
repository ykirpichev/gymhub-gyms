## Hint 1
Create `rng = random.Random(seed)` and draw every random number from it. `rng.randrange(n)` is one uniform draw with replacement. `rng.sample(range(n), k)` is a uniform sample without replacement.

## Hint 2
For weighted draws, reject a negative weight and a non-positive sum. Skip weight 0. Draw index `i` with probability `weight_i / sum(weights)`, independently, `k` times. `k` may be larger than `n`.

## Hint 3
Stratified sampling is without replacement. Give class `c` either `floor(k * n_c / n)` or one more, by largest remainder, so the counts sum to `k` and stay within 1 of the quota. A class with `k * n_c >= n` gets at least one. Shuffle that class and take its count.
