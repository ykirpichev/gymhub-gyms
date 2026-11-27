# Hints for the KV-cache decode step

## Hint 1

Masks: `pos = torch.arange(L, device=...)`; `valid = pos[None, :] < lengths[:, None]` is `[B, L]`. Reshape to `[B, 1, 1, L]` to broadcast against scores of shape `[B, H, 1, L]`.

## Hint 2

Scattering the new token: `rows = torch.arange(B)`; `k[rows, :, lengths] = k_new[:, :, 0]` writes one `[H, Dh]` slab per sequence at its own index. Do it on a clone, or skip it entirely: compute `s_new = (q_new * k_new).sum(-1, keepdim=True) / sqrt(Dh)`, concatenate it to the masked cache scores, softmax, and split the weights back into a `[.., L]` part for `v_cache` and a `[.., 1]` part for `v_new`.

## Hint 3

The separate-new-token version reads each cache once and writes nothing big; that is about as fast as plain PyTorch gets. Beyond that, split-K / flash-decoding (Triton): many programs each handle a block of `L` for one `(b, h)`, produce a partial max, sum and weighted value, and a second step merges them. Programs can stop early at `lengths[b]`, which skips the stale tail of short sequences.
