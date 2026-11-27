#### The idea
A diffusion model trained with the condition randomly dropped can predict noise both with and without the prompt.
Guidance extrapolates from the unconditional prediction toward, and past, the conditional one:
`ε = ε_u + w·(ε_c − ε_u)`. `w > 1` pushes samples harder toward the prompt, at some cost in diversity.

In practice both predictions come from **one** forward pass over a doubled batch, conditional half first. Split it at
the middle of axis 0 and guide.

#### A solution
```python
import numpy as np


def solve(cond, uncond, w):
    scale = float(w)
    if uncond is None:
        batch = np.asarray(cond, dtype=np.float64)
        if batch.ndim == 0 or batch.shape[0] % 2 != 0:
            raise ValueError("a stacked batch must have an even length on axis 0")
        half = batch.shape[0] // 2
        conditional = batch[:half]
        unconditional = batch[half:]
    else:
        conditional = np.asarray(cond, dtype=np.float64)
        unconditional = np.asarray(uncond, dtype=np.float64)
        if conditional.shape != unconditional.shape:
            raise ValueError("cond and uncond must have the same shape")
    return unconditional + scale * (conditional - unconditional)
```

#### What it costs
One elementwise expression, `O(size)`. The real cost is the doubled batch: guidance makes every denoising step
evaluate the model twice. Distilled models (guidance distillation) bake a fixed `w` into a single pass.

#### Pitfalls
- Letting NumPy broadcast mismatched shapes, which silently guides with the wrong tensor.
- Returning both halves of the stacked batch.
- Swapping the halves (conditional is first here; check the convention in any codebase you work in).
- Writing `cond + w·(cond − uncond)`, which is guidance scale `w + 1`.
