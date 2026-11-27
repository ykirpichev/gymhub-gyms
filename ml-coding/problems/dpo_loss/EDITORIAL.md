#### The idea
DPO turns preference data into a classification loss without a separate reward model. The implicit reward of a
response is `β·(log π(y) − log π_ref(y))`, how much more likely the policy makes it than the reference does. The loss
is logistic regression on the reward margin: `−log σ(r_chosen − r_rejected)`.

The numerical part is `log σ(z)` for `|z|` in the thousands. Use
`log σ(z) = min(z, 0) − log(1 + e^(−|z|))`: the exponent is never positive, so nothing overflows, and for very negative
`z` it returns `≈ z` instead of `log(0)`.

#### A solution
```python
import numpy as np


def _log_sigmoid(z):
    # log σ(z) = min(z, 0) - log(1 + exp(-|z|)); the exp never overflows.
    return np.minimum(z, 0.0) - np.log1p(np.exp(-np.abs(z)))


def solve(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    arrays = [np.asarray(v, dtype=np.float64) for v in (policy_chosen, policy_rejected, ref_chosen, ref_rejected)]
    pc, pr, rc, rr = arrays
    if pc.ndim != 1 or pc.size == 0 or any(a.shape != pc.shape for a in arrays) or not beta > 0:
        raise ValueError("expected four equal-length, non-empty 1-D arrays and beta > 0")
    chosen_rewards = beta * (pc - rc)
    rejected_rewards = beta * (pr - rr)
    loss = float(-np.mean(_log_sigmoid(chosen_rewards - rejected_rewards)))
    accuracy = float(np.mean(chosen_rewards > rejected_rewards))
    return loss, chosen_rewards, rejected_rewards, accuracy
```

#### What it costs
`O(N)` given the log-probabilities. The real cost is upstream: four forward passes per pair (policy and reference, on
both responses), or two if the reference log-probs are precomputed once. That is still far cheaper than PPO, which
needs a reward model, a value model and online sampling.

#### Pitfalls
- `np.log(1 / (1 + np.exp(-z)))`: overflow for `z = −1e4` and `-inf` in the loss.
- Using per-token mean log-probs when the method calls for the summed log-prob of the whole response.
- Forgetting the reference model, which removes the KL anchor and lets the policy drift.
- Counting a tie as a correct preference in the accuracy.
