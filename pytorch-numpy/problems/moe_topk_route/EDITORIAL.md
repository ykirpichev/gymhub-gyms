#### The idea
Per token: pick the `k` largest logits (a stable sort of the negated logits breaks ties toward the lower index),
softmax over just those `k` to get the gates, gather the chosen experts' outputs, and take the gate-weighted sum.
`bincount` over the chosen experts gives the load.

#### A solution
```python
import numpy as np


def solve(router_logits, expert_outputs, k):
    logits = np.asarray(router_logits, dtype=np.float64)
    outs = np.asarray(expert_outputs, dtype=np.float64)
    tokens, num_experts = logits.shape
    experts = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    top = np.take_along_axis(logits, experts, axis=1)
    gates = np.exp(top - top.max(axis=1, keepdims=True))
    gates = gates / gates.sum(axis=1, keepdims=True)
    chosen = outs[experts, np.arange(tokens)[:, None]]        # [T, k, D]
    output = (gates[:, :, None] * chosen).sum(axis=1)
    load = np.bincount(experts.ravel(), minlength=num_experts)
    return (output, experts, gates, load)
```
`outs[experts, arange[:, None]]` broadcasts a `T × k` expert index against a `T × 1` token index.

#### In a real layer
Only the routed experts run: tokens are sorted by expert, each expert processes its group as one matmul (see the
GPU gym's `moe_matmul`), and results are scattered back weighted by the gates. The `load` vector feeds the
load-balancing loss, and with expert parallelism it sets how many tokens each GPU receives in the all-to-all.

Some models (Mixtral) softmax over the top `k` as here; others softmax over all experts first and then take the top
`k` without renormalizing. Interviewers like to ask which, because it changes the gate values.

#### Pitfalls
- `np.argsort(-logits)` without `kind="stable"`: ties can come out in either order.
- Softmax over all `E` logits, then picking `k`: different gates, and they do not sum to 1.
