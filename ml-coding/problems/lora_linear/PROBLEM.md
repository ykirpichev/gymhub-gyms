# LoRA linear layer

Write `solve(x, W, b, A, B, alpha)` for a frozen linear layer with a LoRA adapter of rank `r`.

- `x`: `N × in` inputs (list of lists or array). `N ≥ 1`.
- `W`: `out × in` frozen weight. `b`: `out` frozen bias.
- `A`: `r × in`, `B`: `out × r`, the trainable low-rank factors. `r ≥ 1`.
- `alpha`: a float, the LoRA scaling numerator.

With `s = alpha / r`, return a tuple `(y, W_merged, lora_params, full_params)`:

- `y`: `N × out` array, `y = x Wᵀ + b + s · (x Aᵀ) Bᵀ`.
- `W_merged`: `out × in` array, `W + s · B A`. Using it as a plain linear layer, `x W_mergedᵀ + b` gives the same `y`.
- `lora_params`: int, the number of trainable values in `A` and `B`: `r · (in + out)`.
- `full_params`: int, the number of values full fine-tuning would train in the weight: `out · in` (biases are not counted in either).

Raise `ValueError` if any shape is inconsistent: `x`, `W`, `A`, `B` must be 2-D, `b` 1-D, and the `in`, `out` and `r` sizes must agree as above.

The checker compares numbers only. For `y`, the two thin products `(x Aᵀ) Bᵀ` cost about `N · r · (in + out)` multiplications, while building `B A` first costs `out · in · r` before you even touch `x`; `W_merged` is the one place where you do build `B A`. Be ready to say which you would use when, but either order passes.

```
solve(x=[[1, 2]], W=[[1, 0], [0, 1], [1, 1]], b=[0, 0, 1],
      A=[[1, 0], [0, 1]], B=[[1, 0], [0, 1], [0, 0]], alpha=4.0)
-> (array([[3., 6., 4.]]), array([[3., 0.], [0., 3.], [1., 1.]]), 10, 6)
```

Here `r = 2`, so the scale is `4 / 2 = 2`. (At this toy size LoRA trains more values than full fine-tuning; for `in = out = 4096`, `r = 8` it trains 0.4 % of them.)

**Interview focus:** matrix shapes and transposes, the `alpha / r` scaling, counting the parameter savings, and merging the adapter into the base weight.

## This version and the other one

This is the CPU interview version in NumPy: the forward pass, the merged weight and the parameter counts. The LLM gym's `lora_linear` is the GPU version: the forward pass only, on CUDA tensors, with a benchmark that rewards the cheap order of the products.

**Follow-ups** to answer out loud once your solution passes:

- How many trainable parameters does rank 8 add to a 4096 × 4096 layer?
- Why initialize B to zero?
- Merge the adapter into the weight, or keep it separate? When would you do each?
