#### The idea
Any reference point works for `exp`, as long as numerator and denominator use the same one. Keep the sum relative to
the largest value seen so far; when a new chunk raises the max from `m` to `m_new`, every term already in `l` was
`exp(x − m)` and needs to become `exp(x − m_new)`: multiply `l` by `exp(m − m_new)`.

#### A solution
```python
import math


def solve(x, chunk):
    if not x or chunk < 1:
        raise ValueError("need a non-empty x and chunk >= 1")
    m, l = -math.inf, 0.0
    for start in range(0, len(x), chunk):
        part = x[start:start + chunk]
        m_new = max(m, max(part))
        l = l * math.exp(m - m_new) + sum(math.exp(v - m_new) for v in part)
        m = m_new
    return (m, l, [math.exp(v - m) / l for v in x])
```
On the first chunk `m = −∞`, and `exp(−∞) = 0` correctly wipes the empty sum.

#### Why it matters
It turns softmax into one pass, and two partial states `(m₁, l₁)` and `(m₂, l₂)` merge the same way:
`m = max(m₁, m₂)`, `l = l₁·exp(m₁ − m) + l₂·exp(m₂ − m)`. That merge is what lets threads, warps or blocks each
handle part of a row, and it is the core of flash attention, where the output accumulator is rescaled by the same
factor.

#### Pitfalls
- Rescaling by `exp(m_new − m)` (the wrong sign) grows the sum instead of shrinking it.
- Using the final `m` with a sum built against an earlier one: the probabilities no longer sum to 1.
