# Stable softmax and cross-entropy

Write `solve(logits, labels)` for a batch of `N` examples and `C` classes.

- `logits`: an `N × C` list of lists (or array) of floats. `N ≥ 1`, `C ≥ 1`.
- `labels`: `N` integer class indices.

Return a tuple `(probs, loss)`:

- `probs`: an `N × C` NumPy array; row `i` is `softmax(logits[i])`.
- `loss`: a Python float, the **mean** cross-entropy `-(1/N) Σᵢ log probs[i, labels[i]]`.

If any label is outside `[0, C)`, raise `ValueError`.

Logits can be as large as `1e4` or as small as `-1e4`. `exp` of such values overflows or underflows in float64, so a direct implementation returns `nan` or `inf`. Your results must stay finite and accurate.

```
solve([[1.0, 2.0, 3.0]], [2])
-> (array([[0.0900, 0.2447, 0.6652]]), 0.4076)
```

**Interview focus:** subtracting the row max, computing the loss with log-sum-exp instead of `log(softmax)`, shapes, and validating class indices.

## This version and the other one

This is the CPU interview version in NumPy: it returns the probabilities as well as the loss, and its point is the overflow-safe softmax. The LLM gym's `cross_entropy` is the GPU version: CUDA logits over a large vocabulary, padding targets, label smoothing, and a benchmark.

**Follow-ups** to answer out loud once your solution passes:

- Derive the gradient with respect to the logits.
- Add label smoothing. What does it change in the loss and the gradient?
- How would you compute this for a vocabulary of 150K without storing all probabilities?
