# Row softmax

**Level:** medium   **Teaches:** one block per row, shared-memory reductions, numerically stable softmax

## Task

Turn each row of `X` into probabilities: take `exp` of every value, then divide by the row's total. Write the result to `Y`. Every row of `Y` sums to 1.

## Arguments

| Name   | Type          | Shape         | Role   |
| ------ | ------------- | ------------- | ------ |
| `X`    | `const float*` | `rows × cols` | input  |
| `Y`    | `float*`       | `rows × cols` | output |
| `rows` | `int`          |               | size   |
| `cols` | `int`          |               | size   |

Both matrices are row-major. Element `(r, c)` is at `X[r * cols + c]`.

## Formula

For each row `r`, let `m = max over c of X[r, c]`. Then

```
Y[r, c] = exp(X[r, c] - m) / sum over k of exp(X[r, k] - m)
```

Subtracting `m` does not change the answer (it cancels between the top and the bottom), but it keeps `exp` from overflowing. `exp(100)` is `inf` in float32, and `exp(100 - 100)` is 1.

## Example

```
X = [[1, 2, 3]]
m = 3
exp(X - m) = [0.1353, 0.3679, 1.0000]    sum = 1.5032
Y = [[0.0900, 0.2447, 0.6652]]
```

## How to think about it

Each row needs three passes over its values: find the max, add up the exponentials, then write the output. Rows do not depend on each other. Values within a row do, because every output needs the row's max and sum.

That suggests giving each row to a group of threads that cooperate: they split the columns, each thread reduces its share, and the group combines the partial results. One thread per row works too. It is the simplest correct version and a good first step, but it is slow on wide rows.

## In an interview

The bridge from reductions to model kernels: two reductions and an elementwise pass per row, with numerical stability.

Follow-ups to answer out loud once your kernel passes:

- Why subtract the row max? What breaks without it for logits around 100?
- How many times does your kernel read the row from memory, and can you get it to one?
- What is the online softmax, and how does flash attention use it?
