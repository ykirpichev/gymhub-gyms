# Sparse matvec (CSR)

**Level:** medium   **Teaches:** irregular work, warp-per-row, load balancing across rows of very different lengths

## Task

Multiply a sparse matrix `A` by a dense vector `x`. `A` is stored in compressed sparse row (CSR) format: only its nonzero entries are kept, row by row. Write `y = A x`.

## Arguments

| Name      | Type           | Shape      | Role   |
| --------- | -------------- | ---------- | ------ |
| `row_ptr` | `const float*` | `rows + 1` | input  |
| `col_idx` | `const float*` | `nnz`      | input  |
| `values`  | `const float*` | `nnz`      | input  |
| `x`       | `const float*` | `cols`     | input  |
| `y`       | `float*`       | `rows`     | output |
| `rows`    | `int`          |            | size   |
| `cols`    | `int`          |            | size   |
| `nnz`     | `int`          |            | size   |

The nonzeros of row `r` are positions `p` from `row_ptr[r]` to `row_ptr[r + 1] - 1`: entry `values[p]` sits in column `col_idx[p]`. `row_ptr` and `col_idx` hold whole numbers stored as floats; convert them with `(int)`. A row can be empty, and the same column may appear twice in a row (the entries add up).

## Formula

```
y[r] = sum over p in [row_ptr[r], row_ptr[r + 1]) of values[p] * x[col_idx[p]]
```

## Example

```
A = [[1, 0, 0, 2],       row_ptr = [0, 2, 2, 5]
     [0, 0, 0, 0],       col_idx = [0, 3,  1, 2, 3]
     [0, 3, 4, 5]]       values  = [1, 2,  3, 4, 5]

x = [1, 10, 100, 1000]
y = [1 + 2000, 0, 30 + 400 + 5000] = [2001, 0, 5430]
```

## How to think about it

Rows are independent, so one thread per row is correct. But the lengths vary a lot. In the benchmark most rows have a handful of entries, about a third have none, and a few have more than ten thousand. With one thread per row, each warp waits for its longest row, and the 32 threads of a warp read 32 unrelated places of `values` at once, so the loads are not coalesced.

Giving each row a warp fixes both for long rows: the 32 lanes read 32 consecutive nonzeros and sum them with shuffles. For short rows it wastes most of the warp. Fast kernels mix the two, or split the work by nonzeros instead of by rows so every thread gets about the same amount.

There is little arithmetic: two loads and one multiply-add per nonzero, plus a load from `x` that may miss the cache. The benchmark reports bandwidth against the bytes of the matrix and the vectors.

## In an interview

Irregular work: rows of very different lengths. The question is load balance.

Follow-ups to answer out loud once your kernel passes:

- One thread per row, or one warp per row? Which is better for short rows, and for long ones?
- How would you handle a matrix with a few rows of a million entries?
- Where do sparse matrix-vector products appear in ML? (Graph neural networks, sparse attention, pruned weights.)
