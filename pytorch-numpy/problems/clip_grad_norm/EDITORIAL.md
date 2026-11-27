#### The idea
One global norm across every parameter's gradient, as if they were one long vector: the square root of the sum of
all squared elements. If it is above `max_norm`, scale every gradient by the same factor, so the direction of the
update is kept and only its length changes.

#### A solution
```python
import numpy as np


def solve(grads, max_norm):
    arrays = [np.asarray(g, dtype=np.float64) for g in grads]
    norm = float(np.sqrt(sum(float((a * a).sum()) for a in arrays)))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        arrays = [a * scale for a in arrays]
    return (arrays, norm)
```

#### In training
`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` goes after `backward()` and before `optimizer.step()`, and
returns the pre-clipping norm, which is worth logging: spikes in it usually precede loss spikes. With mixed precision
and a gradient scaler, unscale the gradients first (`scaler.unscale_(optimizer)`), or you clip the scaled values.
With sharded training (FSDP), the norm needs an all-reduce of the per-shard sums of squares.

#### Pitfalls
- Clipping each parameter by its own norm: changes the update's direction.
- Returning the norm after clipping.
