## Hint 1

`softmax(x) = exp(x - y[:, None])`, so saving `x` and `y` is enough.

## Hint 2

In `solve`, `torch.tensor(x, dtype=torch.float64, requires_grad=True)` and `y.backward(torch.tensor(grad_y, dtype=torch.float64))`.
