## Hint 1
softmax(x) = softmax(x - c) for any constant c. Pick c = max(x) for each row, so the largest exponent is exp(0) = 1.

## Hint 2
For the loss, avoid log(probs): a tiny probability underflows to 0 and its log is -inf. Use log softmax(x)_k = x_k - logsumexp(x), where logsumexp(x) = m + log Σ exp(x - m) and m = max(x).

## Hint 3
With NumPy: `m = x.max(axis=1, keepdims=True)`, `lse = m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))`, then pick `x[np.arange(N), labels] - lse[:, 0]`.
