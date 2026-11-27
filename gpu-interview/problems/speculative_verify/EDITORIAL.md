#### The idea
Rejection sampling, token by token. Accept the draft's token `x` with probability `min(1, p(x)/q(x))`. On the first
rejection, sample from the normalized residual `max(0, p − q)` and stop. If every draft token survives, the target's
last distribution `p[k]` gives one bonus token for free.

#### A solution
```python
def _sample(d, v):
    total, last = 0.0, None
    for j, prob in enumerate(d):
        if prob > 0:
            last = j
        total += prob
        if v < total:
            return j
    return last


def solve(draft, q, p, u, u_resample):
    out = []
    for i, x in enumerate(draft):
        if q[i][x] <= 0:
            raise ValueError("the draft cannot propose a token it gives probability 0")
        if u[i] < p[i][x] / q[i][x]:
            out.append(x)
            continue
        residual = [max(0.0, a - b) for a, b in zip(p[i], q[i])]
        norm = sum(residual)
        out.append(_sample([r / norm for r in residual], u_resample))
        return out
    out.append(_sample(p[len(draft)], u_resample))
    return out
```

#### Why the output is exact
For a token `y`, the chance it comes out of one position is `q(y)·min(1, p(y)/q(y))` (drafted and accepted) plus the
chance of a rejection times the residual's share of `y`. The two add up to exactly `p(y)`: the draft only changes how
fast tokens come, never their distribution. The speed-up depends on the acceptance rate: with acceptance `α` per
token and `k` drafted, the expected tokens per target pass is `(1 − α^(k+1)) / (1 − α)`.

#### Pitfalls
- Resampling from `p` instead of the residual: biases the output toward tokens the draft already proposed.
- Continuing after a rejection: everything after it was conditioned on a token that is now gone.
- Forgetting the bonus token when all `k` are accepted.
