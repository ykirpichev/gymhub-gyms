# Hints for causal attention

## Hint 1

`scores = q @ k.transpose(-2, -1) / math.sqrt(Dh)` has shape `[B, H, T, T]`. Matmul on 4-D tensors batches over the leading two dimensions.

## Hint 2

`mask = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)` is `True` exactly where `j > i`. `scores.masked_fill(mask, float("-inf"))` broadcasts it over batch and heads. `torch.softmax(scores, dim=-1)` already subtracts the row max.

## Hint 3

To go faster, cut the passes over the `T × T` matrix: scale `q` (a `T × Dh` tensor) instead of the scores, use in-place `masked_fill_` and `softmax` with `out=`, or `torch.baddbmm`. The big win is a fused kernel (Triton) that keeps a running max and sum per query row and never writes the scores to memory, and skips key blocks that lie entirely above the diagonal.
