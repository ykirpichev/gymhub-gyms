#### The idea
Handle the whole batch at once. Score the new query against all `L` cache slots and against the new key; mask the
stale slots (`position >= lengths[b]`) with `-inf`; softmax over the `L + 1` scores; weight the values. The mask comes
from comparing `arange(L)` with `lengths`, so there is no Python loop over sequences.

#### A solution
```python
import math

import torch


def solve(k_cache, v_cache, lengths, q_new, k_new, v_new):
    B, H, L, Dh = k_cache.shape
    scale = 1.0 / math.sqrt(Dh)
    s_cache = (q_new @ k_cache.transpose(-2, -1)) * scale              # [B, H, 1, L]
    stale = torch.arange(L, device=k_cache.device)[None, :] >= lengths[:, None]   # [B, L]
    s_cache = s_cache.masked_fill(stale[:, None, None, :], float("-inf"))
    s_new = (q_new * k_new).sum(-1, keepdim=True) * scale             # [B, H, 1, 1]
    p = torch.softmax(torch.cat([s_cache, s_new], dim=-1), dim=-1)    # [B, H, 1, L + 1]
    return p[..., :L] @ v_cache + p[..., L:] * v_new
```
The new token's own score is always finite, so no row is all `-inf`, even with an empty cache.

#### What it costs
Decoding reads the whole cache for one query per head: `2 · B · H · L · Dh · 4` bytes, 1 GiB for the benchmark, for
almost no math. It is purely memory-bound, which is why KV-cache size (GQA, MQA, FP8 caches, paging) dominates
serving cost. This version reads all `L` slots even when a sequence is short; paged attention kernels read only the
blocks a sequence owns.

#### Pitfalls
- A Python loop over `b` with slicing: correct, and `B` small kernel launches per step.
- Masking with `> lengths` instead of `>=`: slot `lengths[b]` is stale too, since the new token is not written there.
- Multiplying stale scores by 0 instead of masking: `exp(0)` still counts, and stale values around `1e4` leak in.
