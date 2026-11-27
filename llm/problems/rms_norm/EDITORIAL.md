#### The idea
One reduction over the last dimension, kept as a size-1 axis so it broadcasts back: `mean(x²)`, add `eps`, take the
reciprocal square root, multiply. `weight` has shape `[D]` and broadcasts over every leading dimension.

#### A solution
```python
import torch


def solve(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight
```

#### What it costs
Each element is read once and written once: memory-bound. Eager PyTorch launches about five kernels here (`pow`,
`mean`, `add`, `rsqrt`, two `mul`s) and writes temporaries of the full size between some of them, so it moves
several times the minimum bytes. `torch.compile` fuses them into one kernel, and so do the hand-written Triton
and CUDA versions in the GPU gym.

In mixed precision, compute the mean of squares in FP32 (`x.float()`) even when `x` is BF16, then cast back.

#### Pitfalls
- `keepdim=False`: the `[..., ]` result does not broadcast against `[..., D]`.
- `x.norm(dim=-1) / sqrt(D)` is the same quantity but harder to get right with `eps`.
- Modifying `x` in place (`x.mul_`): the task says to leave it unchanged.
