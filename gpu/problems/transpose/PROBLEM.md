# Matrix transpose

**Level:** easy   **Teaches:** coalesced memory access, shared-memory tiles, bank conflicts

## Task

Write the transpose of `A` into `C`. Row `i` of `A` becomes column `i` of `C`.

## Arguments

| Name | Type           | Shape   | Role   |
| ---- | -------------- | ------- | ------ |
| `A`  | `const float*` | `M × N` | input  |
| `C`  | `float*`       | `N × M` | output |
| `M`  | `int`          |         | size   |
| `N`  | `int`          |         | size   |

Both matrices are row-major. Element `(i, j)` of `A` is at `A[i * N + j]`. Element `(j, i)` of `C` is at `C[j * M + i]`.

## Formula

```
C[j, i] = A[i, j]
C[j * M + i] = A[i * N + j]    (flat memory)
```

## Example

```
A = [[1, 2, 3],
     [4, 5, 6]]     M = 2, N = 3

C = [[1, 4],
     [2, 5],
     [3, 6]]
```

## How to think about it

Every output element is a copy of one input element, so there is no arithmetic and no dependence between threads. One thread per element is correct.

The difficulty is memory order. If consecutive threads read along a row of `A`, they write down a column of `C`, and those writes are `M` floats apart. The GPU moves memory in 32-byte sectors, so each scattered 4-byte write costs a whole sector. Either the reads or the writes are strided, whichever way you index the naive kernel.

The fix is to stage a small square tile in shared memory. Read it from `A` along rows, then write it to `C` along rows of `C`. Both global accesses become contiguous, and the reordering happens in fast on-chip memory.

## In an interview

The standard question for coalescing and shared memory. Expect to draw the tile and explain the 33.

Follow-ups to answer out loud once your kernel passes:

- Why is the naive version slow even though it reads every element once?
- Why `[32][33]` and not `[32][32]`? Which accesses conflict without the padding?
- How would you handle a matrix whose sides are not multiples of 32?
