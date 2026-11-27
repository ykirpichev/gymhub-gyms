## Hint 1

Start `m` at `-inf`, `l` at 0 and `acc` at zeros; with `alpha = exp(-inf - m_new) = 0` the first block needs no special case.

## Hint 2

With NumPy, do one query row at a time first; vectorize over rows only after the tests pass.
