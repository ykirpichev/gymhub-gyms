#### The idea
Forward: the stable log-sum-exp, `m + log Σ exp(x − m)`. Backward: the derivative of log-sum-exp is the softmax, and
the softmax can be recovered from what forward saved as `exp(x − y)`, since `y` is the log of the normalizer. Scale it
by the incoming gradient, one value per row.

#### A solution
```python
import torch


class LogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        m = x.max(dim=-1, keepdim=True).values
        y = (m + torch.log(torch.exp(x - m).sum(dim=-1, keepdim=True))).squeeze(-1)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, y = ctx.saved_tensors
        return grad_y.unsqueeze(-1) * torch.exp(x - y.unsqueeze(-1))     # grad_y * softmax(x)


def solve(x, grad_y):
    t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    y = LogSumExp.apply(t)
    y.backward(torch.tensor(grad_y, dtype=torch.float64))
    return (y.detach(), t.grad.detach())
```

#### Checking it
`torch.autograd.gradcheck(LogSumExp.apply, (x,))` compares the backward with finite differences in float64; run it
on any custom `Function`. Saving `y` instead of the softmax costs one value per row instead of a full row: the same
trick flash attention uses when it saves only the log-sum-exp for its backward pass.

#### Pitfalls
- `backward` must return one gradient per input of `forward` (here just `x`), shaped like that input.
- Forgetting `unsqueeze(-1)`: `grad_y` has shape `[N]` and must broadcast across columns.
- Calling `max` without `.values`: `torch.max(dim=...)` returns a pair.
