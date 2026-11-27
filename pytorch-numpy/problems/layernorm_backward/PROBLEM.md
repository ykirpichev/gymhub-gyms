# LayerNorm forward and backward

Layer norm over the last axis of `x` (`N × D`), with learned `gamma` and `beta` (length `D`):

$$\mu = \tfrac{1}{D}\sum_j x_j, \quad \sigma^2 = \tfrac{1}{D}\sum_j (x_j - \mu)^2, \quad \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \odot \hat{x} + \beta.$$

Write `solve(x, gamma, beta, dy, eps)` that returns `(y, dx, dgamma, dbeta)` as NumPy arrays, where `dy = ∂L/∂y`. The row-wise input gradient simplifies to

$$dx = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \Big( g - \operatorname{mean}(g) - \hat{x} \odot \operatorname{mean}(g \odot \hat{x}) \Big), \qquad g = dy \odot \gamma,$$

with the means taken over each row. `dgamma` and `dbeta` sum over rows.

```
solve([[1, 3]], [1, 1], [0, 0], [[1, 0]], 0.0)
-> (array([[-1., 1.]]), array([[0., 0.]]), array([-1., 0.]), array([1., 0.]))
```

(With two features and `eps = 0`, the output is always ±1, so the input gradient is zero.)

#### What interviewers look for

- `dbeta = dy.sum(0)` and `dgamma = (dy * x_hat).sum(0)` straight away.
- A derivation of `dx` through `mu` and `var`, or the compact form above with an explanation of its three terms.
- Keeping `inv_std` from the forward pass instead of recomputing it.

#### Follow-ups

- Check it numerically: how would you write a finite-difference gradient check, and what step size would you use in float64?
- What changes for RMSNorm? (No mean subtraction: one fewer term.)
- Why is layer norm's backward a memory-bound kernel, and which values would a fused kernel save from the forward pass?
