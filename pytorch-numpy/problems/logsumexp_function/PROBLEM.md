# A custom autograd Function

Write a `torch.autograd.Function` subclass `LogSumExp` that computes $y_i = \log \sum_j e^{x_{ij}}$ over the last axis of a 2-D tensor:

- `forward(ctx, x)`: compute it stably (subtract the row max) and save what backward needs.
- `backward(ctx, grad_y)`: return $\partial L/\partial x_{ij} = \text{grad\_y}_i \cdot \mathrm{softmax}(x)_{ij}$.

Then write `solve(x, grad_y)`: make `x` a float64 tensor that requires grad, compute `y = LogSumExp.apply(x)`, call `y.backward(grad_y)`, and return `(y, x.grad)` as tensors (detached).

```
solve([[0.0, 0.0]], [1.0])  ->  (tensor([0.6931]), tensor([[0.5, 0.5]]))
```

#### What interviewers look for

- `ctx.save_for_backward` with the right tensors (the input and the output, or the softmax), and no extra autograd graph inside `backward`.
- The stable forward: `m + log(sum(exp(x - m)))`.
- The backward written from the definition, as a broadcast of `grad_y` over the row.

#### Follow-ups

- How would you check this backward is right? (`torch.autograd.gradcheck` in float64.)
- What should `backward` return for an input that does not need a gradient?
- Save `x` or save the output `y`? What does each cost in memory, and what can you recompute?
