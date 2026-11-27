# AdamW, one step

Write `solve(param, grad, m, v, t, lr, beta1, beta2, eps, weight_decay)` that performs one AdamW step (the rule `torch.optim.AdamW` uses) and returns `(param, m, v)` as NumPy arrays. `param`, `grad`, `m` and `v` are arrays of the same shape; `t` is the step number, starting at 1 for the first step.

```
m  = beta1 * m + (1 - beta1) * grad
v  = beta2 * v + (1 - beta2) * grad ** 2
m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)
param = param - lr * weight_decay * param          # decoupled weight decay, on the old param
param = param - lr * m_hat / (sqrt(v_hat) + eps)
```

```
solve([1.0], [0.5], [0.0], [0.0], 1, 0.1, 0.9, 0.999, 1e-8, 0.0)
-> (array([0.9]), array([0.05]), array([0.00025]))
```

On the first step, bias correction makes `m_hat / sqrt(v_hat)` equal to `sign(grad)`, so every parameter moves by about `lr`.

#### What interviewers look for

- Bias correction, and why it is needed: `m` and `v` start at zero and are biased toward zero for the first steps.
- The difference between Adam with L2 regularization (decay added to the gradient, then rescaled by the adaptive step) and AdamW (decay applied to the weight directly).
- The memory: two extra values per parameter, kept in FP32 in mixed-precision training.

#### Follow-ups

- Which parameters do people usually exclude from weight decay, and why? (Biases and norm weights.)
- What does `eps` do, and what happens with `eps = 1e-8` in FP16?
- How does a learning-rate warmup interact with Adam's early steps?
