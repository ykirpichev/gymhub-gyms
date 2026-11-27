# ReLU

**Level:** easy   **Teaches:** elementwise kernels, memory-bound performance

## Task

Replace every negative value of `A` with 0 and write the result to `C`. Non-negative values pass through unchanged.

## Arguments

| Name | Type           | Shape | Role   |
| ---- | -------------- | ----- | ------ |
| `A`  | `const float*` | `N`   | input  |
| `C`  | `float*`       | `N`   | output |
| `N`  | `int`          |       | size   |

## Formula

```
C[i] = max(A[i], 0)    for i in [0, N)
```

## Example

```
A = [-1, 0, 2]
C = [ 0, 0, 2]
```

## How to think about it

Each output depends only on the input at the same index, so one thread per element is the natural layout. No thread needs anything from another thread.

The work per element is one comparison against 8 bytes of memory traffic (4 read, 4 written). The kernel is memory-bound, so the time is set by how efficiently you move bytes. The branch does not matter much: a `max` compiles to a single instruction with no divergence.

The check is exact (`atol=0`). `max` returns one of its inputs unchanged, so there is no rounding to worry about.

## In an interview

Asked as a warm-up, usually followed by fusion: where would this ReLU live in a real model?

Follow-ups to answer out loud once your kernel passes:

- Fuse ReLU into the matmul or bias add before it. What does that save, in bytes per element?
- Write the backward. What does it need from the forward pass?
- Does a branch (`x > 0 ? x : 0`) cause warp divergence here? (Hint: look at the generated instruction.)
