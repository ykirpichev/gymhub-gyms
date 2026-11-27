#### The idea
Work in log space: `log_softmax` subtracts the row's log-sum-exp, which is computed stably. The target term is a
gather; the smoothing term is a row mean. Build the ignore mask first and gather with a safe index, because `-100`
cannot index the vocabulary.

#### A solution
```python
import torch


def solve(logits, targets, label_smoothing):
    logp = torch.log_softmax(logits, dim=-1)
    keep = targets != -100
    safe = torch.where(keep, targets, torch.zeros_like(targets))
    nll = -logp.gather(1, safe[:, None]).squeeze(1)
    smooth = -logp.mean(dim=-1)
    loss = (1 - label_smoothing) * nll + label_smoothing * smooth
    return (loss * keep).sum() / keep.sum()
```
`F.cross_entropy(logits, targets, ignore_index=-100, label_smoothing=s)` computes the same thing, and is what you
would use outside an exercise.

#### What it costs
The logits dominate: `N × V` floats, 1 GB for 8,192 rows of a 32,000-word vocabulary. `log_softmax` writes another
array of the same size. Fused kernels (as in the GPU gym's `cross_entropy`, or Liger and Apple's cut cross-entropy)
compute the loss row by row without materializing `logp`, and some never materialize the logits either.

#### Pitfalls
- `torch.log(torch.softmax(...))`: the softmax underflows to 0 and the log becomes `-inf` for logits around ±1000.
- Averaging over all rows, padding included.
- Multiplying the mask after a `-inf` has appeared: `0 · inf` is NaN. With the safe index, no `-inf` appears.
