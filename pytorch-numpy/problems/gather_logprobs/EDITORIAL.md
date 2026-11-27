#### The idea
Stable `log_softmax` per row, then pick one entry per row with *paired* integer indexing:
`logp[np.arange(N), targets]` selects `(0, t0), (1, t1), …`.

#### A solution
```python
import numpy as np


def solve(logits, targets):
    x = np.asarray(logits, dtype=np.float64)
    t = np.asarray(targets, dtype=np.int64)
    shifted = x - x.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    picked = logp[np.arange(len(t)), t]
    return (picked, float(-picked.mean()))
```

#### In PyTorch
`torch.log_softmax(x, -1).gather(1, t[:, None]).squeeze(1)`. `gather` is the same paired selection along one axis,
and it is how language-model code picks each token's log-probability (for the loss, or for RL objectives like PPO
and DPO, which need the log-probability of sampled tokens).

#### Pitfalls
- `logp[:, t]`: that is an `N × N` array (every row, every target), not one value per row.
- `log(softmax(x))`: the softmax underflows to 0 for very negative logits, and the log is `-inf`.
