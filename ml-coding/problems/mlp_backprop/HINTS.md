## Hint 1
Keep `z1`, `h` and the softmax probabilities from the forward pass; the backward pass needs all three. Compute the log-softmax as `logits - logsumexp(logits)` after subtracting the row max.

## Hint 2
For a mean cross-entropy, `dlogits = (probs - onehot(y)) / N`. Then `dW2 = h.T @ dlogits` and `db2` is the sum of `dlogits` over the batch axis.

## Hint 3
Go back through the hidden layer: `dh = dlogits @ W2.T`, then `dz1 = dh * (z1 > 0)` (strictly greater, so a zero pre-activation passes no gradient), then `dW1 = X.T @ dz1` and `db1 = dz1.sum(axis=0)`. Check each result's shape against its parameter.
