#### The idea
Update the two moment estimates, correct their start-up bias (they start at zero, so early on they are too small),
apply weight decay directly to the parameter, then take the Adam step.

#### A solution
```python
import numpy as np


def solve(param, grad, m, v, t, lr, beta1, beta2, eps, weight_decay):
    p = np.asarray(param, dtype=np.float64)
    g = np.asarray(grad, dtype=np.float64)
    m = beta1 * np.asarray(m, dtype=np.float64) + (1 - beta1) * g
    v = beta2 * np.asarray(v, dtype=np.float64) + (1 - beta2) * g * g
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    p = p - lr * weight_decay * p                      # decoupled decay, on the old value
    p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
    return (p, m, v)
```

#### Adam versus AdamW
Adam with L2 regularization adds `wd · p` to the *gradient*, so the decay is divided by `√v̂` like everything else:
parameters with large gradients are barely decayed. AdamW applies the decay to the parameter directly, the same for
every weight, which is what "weight decay" was supposed to mean. That difference is the paper (Loshchilov and
Hutter) and a favourite interview question.

Memory: `m` and `v` are two extra FP32 values per parameter, 8 bytes, which is why mixed-precision Adam needs about
16 bytes per parameter in total.

#### Pitfalls
- Starting `t` at 0: `1 − beta ** 0` is 0, a division by zero.
- Bias-correcting `m` and `v` in place, so the next step starts from the corrected values.
- Putting `eps` inside the square root: a different (also common) variant, but not PyTorch's.
