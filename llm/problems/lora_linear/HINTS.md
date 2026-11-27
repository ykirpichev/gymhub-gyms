# Hints for LoRA linear

## Hint 1

Track shapes: `x @ W.T` is `[N, out]`; `x @ A.T` is `[N, r]`; `[N, r] @ B.T` is `[N, out]`. The two `[N, out]` results add, and `b` broadcasts over rows.

## Hint 2

Matrix products are associative but not equally expensive. Count multiply-adds for `(x A^T) B^T` versus `x (B A)^T`. For the benchmark, `N = in = out = 4096`, `r = 16`.

## Hint 3

`torch.addmm(b, x, W.T)` fuses the bias into the matmul. You can also fuse the scale and the add: `torch.addmm(base, x @ A.T, B.T, alpha=alpha / r)` computes `base + (alpha / r) * (x A^T) B^T` in one call.
