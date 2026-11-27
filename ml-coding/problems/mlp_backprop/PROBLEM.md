# MLP forward and backward pass

Write `solve(X, y, W1, b1, W2, b2)` that runs a one-hidden-layer network forward on a batch and backpropagates the loss by hand.

- `X`: an `N × D` list of lists (or array) of floats, `N ≥ 1`.
- `y`: `N` integer class labels in `[0, C)`.
- `W1`: `D × H`, `b1`: length `H`, `W2`: `H × C`, `b2`: length `C`, all floats.

The network and the loss:

```
z1     = X @ W1 + b1          # N × H
h      = ReLU(z1)             # max(z1, 0)
logits = h @ W2 + b2          # N × C
loss   = -(1/N) Σᵢ log softmax(logits[i])[y[i]]
```

Use the derivative `ReLU'(z) = 1` for `z > 0` and `0` for `z <= 0`, so `ReLU'(0) = 0`. Logits can be in the thousands, so the loss must be computed stably (it stays finite).

Return a tuple `(loss, dW1, db1, dW2, db2)`: `loss` is a float, and each gradient is a NumPy array with the **same shape as its parameter**, holding the derivative of `loss` with respect to it.

```
solve(X=[[1, 2], [-1, 0.5]], y=[0, 1],
      W1=[[0.5, -1], [0.25, 1]], b1=[0, 0.5],
      W2=[[1, -1], [0.5, -0.25]], b2=[0, 0])
-> loss 0.872206, db2 = array([0.387743, -0.387743]),
   db1 = array([-0.042088, 0.290808]), dW1 = array([[-0.042088, -0.322373], [-0.084175, 0.121729]])
```

Here `z1 = [[1, 1.5], [-0.375, 2]]`: hidden unit 0 is off for the second sample, so that sample adds nothing to `dW1[:, 0]` or `db1[0]`.

**Interview focus:** caching the forward values, the softmax cross-entropy gradient `(softmax - onehot) / N`, gradient flow through the ReLU mask, bias gradients as sums over the batch, and matching every gradient's shape to its parameter.

**Follow-ups** to answer out loud once your solution passes:

- How would you check your gradients numerically?
- What changes with a sigmoid hidden layer instead of ReLU, and why do deep sigmoid networks train slowly?
- How does the memory for activations scale with batch size and depth?
