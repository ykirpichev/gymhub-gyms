## Hint 1
`temperature == 0` is `argmax`, with the lowest index winning a tie. Do not divide by zero. For a positive temperature, subtract the max of `logits / temperature` before `exp`.

## Hint 2
Rank with the lower index first on a tie: `np.lexsort((np.arange(V), -probs))`. After top-k, set the dropped tokens to 0 and divide by the new sum **before** top-p. A nucleus computed on the unnormalized top-k masses keeps a different set.

## Hint 3
Top-p walks that ranked list and stops at the first prefix whose cumulative probability is at least `p`, always keeping the first token. Zero the rest, renormalize, then take the smallest index `i` with `cumsum(probs)[i] > u` (`np.searchsorted(..., side="right")`). A leading zero must not be selected when `u` is 0.
