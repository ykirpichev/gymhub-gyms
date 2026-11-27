#### The idea
One batched matmul gives every head's `T × T` scores; mask the upper triangle with `-inf`; softmax over the last
axis; one more batched matmul with `v`.

#### A solution
```python
import math

import torch


def solve(q, k, v):
    T, Dh = q.shape[-2], q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)                    # [B, H, T, T]
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
```
In real code: `F.scaled_dot_product_attention(q, k, v, is_causal=True)`, which dispatches to a flash or
memory-efficient kernel.

#### What it costs
`4 · B · H · T² · Dh` FLOPs for the two matmuls. The score matrix is the problem: `B · H · T²` floats, 256 MiB for
the benchmark's `4 × 16 × 1024²`, written by the first matmul, read and written by the mask and the softmax, read
again by the second matmul. A fused kernel (flash attention) never stores it, which is why it is several times faster
and why its memory grows linearly in `T`.

#### Pitfalls
- `diagonal=0` masks the diagonal too: row 0 is then all `-inf` and the softmax returns NaN.
- Masking with `-1e9` works in FP32 but overflows in FP16; `-inf` (or the dtype's minimum) is safe.
- Scaling after the softmax instead of before.
