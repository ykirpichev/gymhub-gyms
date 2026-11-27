# Cross-entropy

**Level:** medium   **Teaches:** the log-sum-exp trick, online max and sum in one pass, very long rows

## Task

A language model's training loss compares each row of logits with its target token. For each row `r`, compute the negative log-probability of token `targets[r]` under the softmax of `logits[r, :]` and write it to `loss[r]`. This is `torch.nn.functional.cross_entropy(logits, targets, reduction="none")`.

## Arguments

| Name      | Type           | Shape      | Role   |
| --------- | -------------- | ---------- | ------ |
| `logits`  | `const float*` | `rows × V` | input  |
| `targets` | `const float*` | `rows`     | input  |
| `loss`    | `float*`       | `rows`     | output |
| `rows`    | `int`          |            | size   |
| `V`       | `int`          |            | size   |

`targets` holds whole numbers in `[0, V)` stored as floats. `logits` is row-major: `(r, j)` is at `logits[r * V + j]`. `V` is the vocabulary size, up to about 50,000 in the tests.

## Formula

```
lse[r]  = log(sum over j of exp(logits[r, j]))
loss[r] = lse[r] - logits[r, targets[r]]
```

Computed as written, `exp` overflows to `inf` once a logit passes about 88. Subtract the row's maximum `m` first: `lse = m + log(sum over j of exp(logits[r, j] - m))`. The tests include rows with logits near `1000` and near `-1000`.

## Example

```
logits  = [[1, 2, 3],      targets = [2, 0]
           [0, 0, 0]]

row 0: m = 3, sum = e^-2 + e^-1 + e^0 = 1.5032, lse = 3.4076, loss = 3.4076 - 3 = 0.4076
row 1: lse = log 3 = 1.0986,                                   loss = 1.0986 - 0 = 1.0986
loss = [0.4076, 1.0986]
```

## How to think about it

This is `row_softmax` with a twist: rows are very long and the output is one number per row, so you never need the probabilities themselves. A simple kernel makes two passes over the row, one for the maximum and one for the sum of exponentials. That reads the logits twice, and the logits are almost all of the memory traffic.

An online version keeps a running maximum `m` and a running sum `s` of `exp(x - m)`. When a new value `x` exceeds `m`, rescale the sum: `s = s * exp(m - x) + 1` and `m = x`. Two partial states `(m1, s1)` and `(m2, s2)` combine the same way: `m = max(m1, m2)`, `s = s1 * exp(m1 - m) + s2 * exp(m2 - m)`. That is one pass, and it is exactly the trick flash attention uses for its softmax (see `causal_attention`).

## In an interview

A row-wise log-sum-exp plus a gather; often asked together with its backward and with very large vocabularies.

Follow-ups to answer out loud once your kernel passes:

- Why compute the loss from logits with log-sum-exp instead of from softmax probabilities?
- What is the gradient with respect to the logits, and can you produce it in the same kernel?
- With a 150K vocabulary and 8K tokens, the logits alone are 4.9 GB in FP32. How do fused or chunked losses avoid storing them?

## This version and the other one

This is the kernel version: one loss per row from a raw `float*` in CUDA or Triton, with a benchmark against memory bandwidth. The LLM gym's `cross_entropy` is the PyTorch version: padding targets of `-100`, label smoothing and a mean over rows.
