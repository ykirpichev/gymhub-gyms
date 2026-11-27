## Hint 1

Compute `x_hat` and `inv_std = 1 / sqrt(var + eps)` per row with `keepdims=True`.

## Hint 2

`g = dy * gamma`, then `dx = inv_std * (g - g.mean(1, keepdims=True) - x_hat * (g * x_hat).mean(1, keepdims=True))`.
