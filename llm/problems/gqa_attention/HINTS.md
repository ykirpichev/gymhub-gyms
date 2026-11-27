# Hints for grouped-query attention

## Hint 1

Which query heads share KV head `g`? Heads `g·G` to `g·G + G − 1`, a contiguous block, so `q.view(B, Hkv, G, L, Dh)` groups them.

## Hint 2

`k.unsqueeze(2)` is `[B, Hkv, 1, L, Dh]`. `qg @ k.unsqueeze(2).transpose(-2, -1)` broadcasts over the group dimension and gives scores `[B, Hkv, G, L, L]`.

## Hint 3

Build the causal mask once with `torch.ones(L, L, dtype=torch.bool).triu(1)` on the right device, `masked_fill` with `-inf`, softmax, multiply by `v.unsqueeze(2)`, and view the result back to `[B, Hq, L, Dh]`.
