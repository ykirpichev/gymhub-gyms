# Hints for paged attention decode

## Hint 1

`k_pool[block_table]` indexes the first dimension with a `[B, M]` tensor and gives `[B, M, S, H, Dh]`: every sequence's blocks, in logical order.

## Hint 2

`.reshape(B, M * S, H, Dh)` turns (block, slot) into one token index `t = j·S + slot`. Permute to `[B, H, M·S, Dh]` so heads come before tokens.

## Hint 3

Scores are `(k * q[:, :, None, :]).sum(-1)` or an `einsum("bhd,bhtd->bht", ...)`. Mask with `torch.arange(M * S)[None, :] >= lengths[:, None]`, broadcast over heads, then softmax and weight the values the same way.
