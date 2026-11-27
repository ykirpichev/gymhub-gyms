# Embedding lookup

**Level:** easy   **Teaches:** gathers, one block per output row, coalescing along the row

## Task

A language model starts by turning each token id into a vector. For each `i`, copy row `ids[i]` of `table` into row `i` of `out`. This is `torch.nn.functional.embedding`.

## Arguments

| Name    | Type           | Shape   | Role   |
| ------- | -------------- | ------- | ------ |
| `ids`   | `const float*` | `N`     | input  |
| `table` | `const float*` | `V × D` | input  |
| `out`   | `float*`       | `N × D` | output |
| `N`     | `int`          |         | size   |
| `V`     | `int`          |         | size   |
| `D`     | `int`          |         | size   |

`ids` holds whole numbers in `[0, V)` stored as floats. Convert one with `(int)ids[i]`. Both matrices are row-major: `table[v, d]` is at `table[v * D + d]`.

## Formula

```
out[i, d] = table[ids[i], d]
```

## Example

```
ids   = [2, 0, 2]
table = [[ 0,  1],
         [10, 11],
         [20, 21]]

out   = [[20, 21],
         [ 0,  1],
         [20, 21]]
```

## How to think about it

There is no arithmetic at all. The kernel is a copy, so its speed is how close it gets to the memory bandwidth.

The rows of `table` that you read are scattered, but each row is contiguous. Neighbouring threads should therefore work on neighbouring `d` inside the same row, not on neighbouring `i`. If thread `t` handled token `t` and looped over `d`, the 32 threads of a warp would read 32 different rows at once and every load would touch a different cache line.

The same gather appears inside mixture-of-experts layers, where tokens are grouped by the expert they are routed to (see `moe_matmul`).

## In an interview

A gather: each output row copies one table row. The question is about coalescing and the backward pass.

Follow-ups to answer out loud once your kernel passes:

- Map threads so a warp copies one row with coalesced loads. What if the row is 4096 floats?
- Write the backward. Why does it need atomics or a sort when token ids repeat?
- How is a vocabulary of 150K tokens split across GPUs in tensor parallelism?
