#### The idea
Two thin matmuls for the adapter path, never the full `B A` matrix: `(x Aᵀ)` is `[N, r]`, then times `Bᵀ` is
`[N, out]`. Add the frozen projection and bias, and scale the adapter by `alpha / r`.

#### A solution
```python
import torch


def solve(x, W, b, A, B, alpha):
    r = A.shape[0]
    return torch.addmm(b, x, W.t()) + (alpha / r) * ((x @ A.t()) @ B.t())
```
`torch.addmm(b, x, W.t())` folds the bias into the matmul call.

#### What it costs
The frozen projection costs `N · in · out` multiply-adds. The adapter path costs `N · r · (in + out)`: for
`r = 16`, `in = out = 4096` that is 0.8 percent extra. Forming `B A` first costs `out · in · r` work plus an
`[out, in]` temporary on every call, as much memory traffic as `W` itself.

During training only `A` and `B` get gradients and optimizer state, which is where LoRA's memory saving comes from.
For inference, merge once: `W' = W + (alpha / r) B A`, and the layer is an ordinary linear layer again.

#### Pitfalls
- Scaling by `alpha` instead of `alpha / r`.
- `x @ (B @ A).t()`: right answer, wrong cost, and the benchmark shows it.
- Transposes: `W` is `[out, in]` as in `nn.Linear`, so the product is `x Wᵀ`.
