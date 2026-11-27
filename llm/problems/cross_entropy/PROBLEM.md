# Cross-entropy loss

**Level:** easy   **Teaches:** log-softmax, the log-sum-exp trick, masked means, label smoothing

## Task

Compute the language-model training loss: the mean negative log-likelihood of each row's target token under the softmax of its logits. Rows whose target is `-100` are padding and do not count, neither in the sum nor in the count. With `label_smoothing = s > 0`, the target distribution puts `1 - s` on the true token and spreads `s` evenly over all `V` tokens. Return a 0-dim float32 tensor.

## Arguments

| Name              | Type           | Shape    | Role                                         |
| ----------------- | -------------- | -------- | -------------------------------------------- |
| `logits`          | `torch.Tensor` | `[N, V]` | unnormalized scores, float32, CUDA           |
| `targets`         | `torch.Tensor` | `[N]`    | int64 token ids in `[0, V)`, or `-100`       |
| `label_smoothing` | `float`        |          | `s` in `[0, 1)`                              |
| returns           | `torch.Tensor` | `[]`     | mean loss over the rows that are not ignored |

At least one row is never ignored.

## Formula

```
logp[i, j] = logits[i, j] - logsumexp_j(logits[i, :])
nll[i]     = -logp[i, targets[i]]
smooth[i]  = -mean_j logp[i, j]
loss[i]    = (1 - s) * nll[i] + s * smooth[i]
result     = sum of loss[i] over kept rows / number of kept rows
             (kept rows: targets[i] != -100)
```

## Example

```
logits  = [[2, 1, 0],      targets = [0, -100]   s = 0
           [0, 0, 0]]
row 0: logsumexp = log(e^2 + e^1 + e^0) = 2.4076,  nll = 2.4076 - 2 = 0.4076
row 1: ignored
result  = 0.4076
```

## How to think about it

Never compute `exp(logits)` directly: logits of size 1000 overflow float32. Subtract the row max first (that is what `logsumexp` does). Once you have `logp`, the true-token term is a gather and the smoothing term is a row mean. Build the ignore mask before the gather, because `-100` is not a valid index. Do not call `torch.nn.functional.cross_entropy`, `nll_loss`, or `log_softmax`; build it from tensor operations.

## Tests

Shapes from `[1, 2]` to `[8, 32000]`, ignored rows, label smoothing, and a case where only one row of 64 counts and the logits are around ±1000. Benchmark input: `logits[8192, 32000]`, `label_smoothing = 0.1`.

## This version and the other one

This is the GPU version: CUDA logits over a large vocabulary, padding targets of `-100`, label smoothing, a 0-dim result, and a benchmark. ML coding's `softmax_cross_entropy` is the CPU interview version in NumPy: it returns the probabilities and the mean loss, and its point is the overflow-safe softmax.
