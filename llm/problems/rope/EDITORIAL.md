#### The idea
Build the angle table once, `[T, Dh/2]`, as an outer product of positions and inverse frequencies. Duplicate it to
`[T, Dh]` for the "rotate half" layout, then `out = x · cos + rotate_half(x) · sin`, broadcasting over batch and heads.

#### A solution
```python
import torch


def solve(q, start):
    B, H, T, Dh = q.shape
    half = Dh // 2
    inv_freq = 10000.0 ** (-torch.arange(half, device=q.device, dtype=torch.float32) * 2 / Dh)
    pos = torch.arange(start, start + T, device=q.device, dtype=torch.float32)
    theta = pos[:, None] * inv_freq[None, :]                 # [T, half]
    cos = torch.cat([theta.cos(), theta.cos()], dim=-1)      # [T, Dh], broadcasts over B and H
    sin = torch.cat([theta.sin(), theta.sin()], dim=-1)
    rotated = torch.cat([-q[..., half:], q[..., :half]], dim=-1)
    return q * cos + rotated * sin
```

#### Why it works
Rotating a query at position `p` and a key at position `p'` by angles `pω` and `p'ω` leaves their dot product a
function of `p − p'` only: attention sees *relative* positions without any extra parameters. Low-index pairs spin
fast (fine-grained position), high-index pairs slowly (long range). Context-extension tricks (NTK scaling, YaRN)
change `inv_freq` so that long positions map back into the range seen in training.

In a real model the `cos`/`sin` tables are cached and shared by every layer; during decoding, `start` is the cache
length, which is why the exercise takes it.

#### Pitfalls
- Building positions from 0 instead of `start`: correct only for prefill.
- The pairing: `(i, i + Dh/2)`, not `(2i, 2i + 1)`. Mixing layouts between a checkpoint and the code is a classic
  silent bug: the model still runs, just badly.
- Computing angles in FP16: positions in the thousands times high frequencies lose all their precision.
