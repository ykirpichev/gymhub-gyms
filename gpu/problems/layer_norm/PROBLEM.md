# Layer norm

**Level:** medium   **Teaches:** one block per row, mean and variance reductions, two-pass vs Welford

## Task

For each row of `X`, subtract the row's mean and divide by its standard deviation. Then multiply column `c` by `gamma[c]` and add `beta[c]`. Write the result to `Y`. This is `torch.nn.functional.layer_norm` over the last dimension.

## Arguments

| Name    | Type           | Shape         | Role   |
| ------- | -------------- | ------------- | ------ |
| `X`     | `const float*` | `rows × cols` | input  |
| `gamma` | `const float*` | `cols`        | input  |
| `beta`  | `const float*` | `cols`        | input  |
| `Y`     | `float*`       | `rows × cols` | output |
| `rows`  | `int`          |               | size   |
| `cols`  | `int`          |               | size   |
| `eps`   | `float`        |               | scalar |

Both matrices are row-major. Element `(r, c)` is at `X[r * cols + c]`. `gamma` and `beta` are shared by all rows.

## Formula

For each row `r`:

```
mean = (1 / cols) * sum over c of X[r, c]
var  = (1 / cols) * sum over c of (X[r, c] - mean)^2
Y[r, c] = (X[r, c] - mean) / sqrt(var + eps) * gamma[c] + beta[c]
```

The variance divides by `cols`, not `cols - 1`. `eps` keeps the division safe when a row is constant (then `var` is 0 and `Y[r, c] = beta[c]`).

## Example

```
X     = [[1, 2, 3, 4]]
gamma = [1, 1, 2, 2]     beta = [0, 0, 0, 1]     eps = 1e-5

mean = 2.5    var = 1.25    sqrt(var + eps) = 1.1180
normalized = [-1.3416, -0.4472, 0.4472, 1.3416]
Y     = [[-1.3416, -0.4472, 0.8944, 3.6833]]
```

## How to think about it

Rows are independent. Within a row, every output needs two numbers that depend on the whole row: the mean and the variance. So the work per row is: reduce to get the mean, reduce again to get the variance, then one elementwise pass to write `Y`.

One thread per row is the simplest correct version. It is slow because a row of 4096 values is handled by a single thread. Giving each row to a whole block lets the threads split the columns and combine their partial sums, the same pattern as row softmax.

Watch the precision. `var = E[x^2] - mean^2` needs only one pass, but it subtracts two large, close numbers when the mean is far from zero. Computing the mean first and then summing `(x - mean)^2` avoids that. Welford's method gets both in one pass and stays accurate.

## In an interview

Asked as a fused, memory-bound kernel: one block per row, mean and variance, then normalize.

Follow-ups to answer out loud once your kernel passes:

- Compute mean and variance in one pass. Which formula is numerically safe? (Welford, or a two-pass in registers.)
- What does the backward need from the forward pass, to avoid recomputing?
- What changes for RMSNorm, and why do recent LLMs prefer it?
