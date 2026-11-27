# Hints for the GRPO loss

## Hint 1

`groups = rewards.view(-1, G)`; `groups.mean(1, keepdim=True)` and `groups.std(1, keepdim=True)` (PyTorch's default `std` divides by `G − 1`). Flatten the advantages back to `[N]`.

## Hint 2

Broadcast the advantage over tokens with `A[:, None]`. `torch.clamp(ratio, 1 - eps, 1 + eps)` and `torch.minimum` give the clipped surrogate.

## Hint 3

Masked mean per row: `(token_loss * m).sum(1) / m.sum(1)` with `m = mask.float()`, or `torch.where(mask, token_loss, 0)` to be safe if masked values could be huge. Then `.mean()` over rows.
