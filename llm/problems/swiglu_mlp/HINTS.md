# Hints for the SwiGLU feed-forward block

## Hint 1

`x @ w_gate.T` works for any leading shape: matmul broadcasts over everything but the last two dimensions.

## Hint 2

`torch.sigmoid` is stable for any input. `z * torch.sigmoid(z)` never overflows, unlike `z / (1 + torch.exp(-z))` for very negative `z`.

## Hint 3

Stack the gate and up weights once: `w = torch.cat([w_gate, w_up])` is `[2F, D]`; `x @ w.T` gives both projections, and `.chunk(2, dim=-1)` splits them. In a real model you would store the weight already stacked.
