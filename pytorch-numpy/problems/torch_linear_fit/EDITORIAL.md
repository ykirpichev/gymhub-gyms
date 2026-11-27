#### The idea
The standard loop, in the standard order: zero the gradients, forward, loss, backward, step. Record the loss before
the step, since that is the loss the gradients came from.

#### A solution
```python
import torch


def solve(X, y, w0, b0, lr, steps):
    X = torch.tensor(X, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    w = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(float(b0), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([w, b], lr=lr)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = ((X @ w + b - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return (w.detach(), b.item(), losses)
```

#### Check the first step by hand
For the example, the prediction is 0, the errors are `[−2, −4]` and the loss is 10. The gradients are
`∂L/∂w = mean(2·err·x) = −10` and `∂L/∂b = mean(2·err) = −6`, so one step with `lr = 0.1` gives `w = 1`, `b = 0.6`.
Doing one step by hand like this is the fastest way to find a sign or a factor-of-two bug.

#### Pitfalls
- Skipping `zero_grad()`: gradients accumulate across steps and the updates grow.
- `losses.append(loss)` without `.item()`: keeps every step's graph alive and leaks memory.
- Creating the tensors with the default float32 dtype; the task asks for float64.
