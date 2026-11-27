## Hint 1
`np.swapaxes(k, -1, -2)` transposes only the last two axes, so `q @ np.swapaxes(k, -1, -2)` works for any number of batch dimensions. Divide by `sqrt(d_k)`, the last dimension of `q`, not of `v`.

## Hint 2
Masking after the softmax leaves rows that no longer sum to 1. Instead set masked scores to `-inf` with `np.where(mask, scores, -np.inf)`; `exp(-inf)` is exactly 0.

## Hint 3
Subtract each row's max before `exp`. In a fully masked row the max is `-inf`, and `-inf - (-inf)` is `nan`: replace a non-finite max by 0 so that row's `exp` is all zeros, and divide by the row sum only where it is non-zero.
