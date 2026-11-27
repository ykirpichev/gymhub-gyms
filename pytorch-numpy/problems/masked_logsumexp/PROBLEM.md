# Stable log-sum-exp with masks

Write `solve(x, mask)` for a 2-D array `x` and a boolean array `mask` of the same shape. Entries where `mask` is false are masked out: set them to `-inf` first, as an attention kernel does with a causal mask, then return the log-sum-exp of each row as a NumPy array, computed stably:

$$\operatorname{lse}(x) = m + \log \sum_j e^{x_j - m}, \qquad m = \max_j x_j.$$

A row whose entries are all masked has log-sum-exp `-inf`, without warnings or NaN.

```
solve([[0, 0], [1000, 1000]], [[True, True], [True, True]])  ->  array([0.6931, 1000.6931])
solve([[1, 2]], [[False, False]])                            ->  array([-inf])
```

#### What interviewers look for

- Subtracting the max, and handling `m = -inf`, where `x - m` is `-inf - (-inf) = nan`.
- Knowing where this shows up: softmax denominators, cross-entropy, attention with a causal mask, the `lse` saved by flash attention.

#### Follow-ups

- What should the softmax of a fully masked row be? (Attention kernels usually output zeros.)
- What does `torch.logsumexp` do with an all `-inf` row?
- How would you compute this in one pass over a long row, as a stream?
