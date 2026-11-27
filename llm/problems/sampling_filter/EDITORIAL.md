#### The idea
Top-k: everything below the `k`-th largest value per row becomes `-inf`. Top-p: sort the survivors descending,
softmax, and keep a token while the probability mass strictly *before* it is below `top_p`. That mask is in sorted
order; `scatter` puts it back in vocabulary order.

#### A solution
```python
import torch


def solve(logits, top_k, top_p):
    x = logits.clone()
    V = x.shape[-1]
    if 0 < top_k < V:
        kth = torch.topk(x, top_k, dim=-1).values[:, -1:]
        x = x.masked_fill(x < kth, float("-inf"))
    if top_p < 1.0:
        sorted_x, order = torch.sort(x, dim=-1, descending=True)
        p = torch.softmax(sorted_x, dim=-1)
        before = torch.cumsum(p, dim=-1) - p                 # mass strictly ahead
        drop_sorted = before >= top_p
        drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)
        x = x.masked_fill(drop, float("-inf"))
    return x
```
The best token has `before = 0`, so it always survives.

#### What it costs
Sorting a 128,000-wide row per sequence is the expensive part. When `top_k` is set, it is cheaper to sort only the
`top_k` survivors (`torch.topk` returns them sorted) and apply top-p within them; production samplers (vLLM,
FlashInfer) go further with rejection-based sampling that avoids the sort.

#### Pitfalls
- Applying top-p to the original distribution instead of the one renormalized after top-k.
- Using the inclusive cumulative sum: it drops the token that crosses the threshold, and can drop the best token
  when it alone exceeds `top_p`.
- Forgetting `clone()`: `masked_fill` returns a new tensor, but in-place variants would modify the caller's logits.
