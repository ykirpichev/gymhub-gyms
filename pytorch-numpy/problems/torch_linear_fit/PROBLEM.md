# Fit a linear model with autograd

Write `solve(X, y, w0, b0, lr, steps)` that fits `y ≈ X @ w + b` by full-batch gradient descent in PyTorch:

- `X`: `N × D`, `y`: `N`, `w0`: `D` initial weights, `b0`: initial bias (a float). Use float64.
- Each step: zero the gradients, compute the mean squared error `((X @ w + b - y) ** 2).mean()`, backpropagate, and update with `torch.optim.SGD(lr=lr)` (no momentum).

Return `(w, b, losses)`: the final weights (a tensor), the final bias (a float) and the list of `steps` loss values, each recorded **before** that step's update.

```
solve([[1.0], [2.0]], [2.0, 4.0], [0.0], 0.0, 0.1, 1)
-> (tensor([1.]), 0.6, [10.0])
```

#### What interviewers look for

- Parameters as leaf tensors with `requires_grad=True` (or `nn.Parameter`), and an optimizer built over them.
- `zero_grad`, forward, loss, `backward`, `step`, in order, and `loss.item()` to record a Python float without keeping the graph alive.

#### Follow-ups

- Replace the hand-made parameters with `nn.Linear`. What does its default initialization do?
- Switch to mini-batches with a `DataLoader`. What changes in the loss curve?
- Why is calling `.item()` every step slow on a GPU, and what would you do instead?
