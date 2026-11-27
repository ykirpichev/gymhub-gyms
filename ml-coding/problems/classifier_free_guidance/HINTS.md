## Hint 1
The baseline is the unconditional prediction: `uncond + w * (cond - uncond)`. That is `w * cond + (1 - w) * uncond`. `w = 0` must return `uncond`, not `cond`.

## Hint 2
Require `cond.shape == uncond.shape`. NumPy would happily broadcast `(2, 1)` against `(2, 3)`; this exercise rejects that.

## Hint 3
When `uncond` is `None`, split axis 0 in half. `batch[:half]` is conditional and `batch[half:]` is unconditional. Guide them and return the half, not the stacked batch. An odd length on axis 0 is a `ValueError`.
