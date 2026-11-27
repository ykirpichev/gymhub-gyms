# Online softmax

A safe softmax subtracts the row maximum before exponentiating, which seems to need two passes: one for the max, one for the sum. The online version keeps a running max `m` and a running sum `l` of `exp(x − m)`, and rescales `l` whenever `m` grows:

```
m = -inf, l = 0
for each chunk c:
    m_new = max(m, max(c))
    l = l * exp(m - m_new) + sum(exp(c_j - m_new) for c_j in c)
    m = m_new
softmax(x)_j = exp(x_j - m) / l
```

Write `solve(x, chunk)` that processes the list `x` in consecutive chunks of `chunk` elements (the last one may be shorter) and returns a tuple `(m, l, probs)`: the final max, the final sum, and the softmax of `x` as a list.

```
solve([1, 2, 3], 2)  ->  (3, 1.50321, [0.09003, 0.24473, 0.66524])
```

Raise `ValueError` if `x` is empty or `chunk < 1`.

#### What interviewers look for

- The rescale factor `exp(m_old − m_new)`, and why it is always ≤ 1, so nothing overflows.
- That the result is exactly the two-pass softmax, whatever the chunk size.
- Connecting it to flash attention, which applies the same rescale to a running weighted sum of value rows.

#### Follow-ups

- Two GPU threads each processed half the row and hold `(m1, l1)` and `(m2, l2)`. How do you combine them? (This is how a warp reduces an online softmax.)
- Why is `m = -inf` safe at the start, and what goes wrong if a whole row is `-inf` (a fully masked row in attention)?
- Where does this save memory traffic in a GPU kernel, compared with the two-pass version?
