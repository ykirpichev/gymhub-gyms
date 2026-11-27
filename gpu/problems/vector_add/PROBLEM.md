# Vector add

**Level:** easy   **Teaches:** one thread per element, grid and block indexing, bounds checks

## Task

Add two vectors element by element and write the result to `C`.

## Arguments

| Name | Type           | Shape | Role   |
| ---- | -------------- | ----- | ------ |
| `A`  | `const float*` | `N`   | input  |
| `B`  | `const float*` | `N`   | input  |
| `C`  | `float*`       | `N`   | output |
| `N`  | `int`          |       | size   |

## Formula

```
C[i] = A[i] + B[i]    for i in [0, N)
```

## Example

```
A = [1, 2, 3]
B = [4, 5, 6]
C = [5, 7, 9]
```

## How to think about it

Every output depends on one element of `A` and one of `B`, and on nothing else. There is nothing to share, so give each element its own thread.

The launch rounds the thread count up to a whole number of blocks, so the last block usually has threads past the end of the arrays. Those threads must do nothing.

There is almost no arithmetic: one add per 12 bytes moved. The speed limit is memory bandwidth, not compute. A good kernel reads and writes each byte once, with neighbouring threads touching neighbouring addresses.

## In an interview

The opener of many kernel rounds: write it on paper with the index, the bounds check and the launch configuration, then say why it can't beat memory bandwidth.

Follow-ups to answer out loud once your kernel passes:

- What is the arithmetic intensity, and the lowest time for 2^28 floats at 3 TB/s?
- How would you size the grid: one thread per element, or a grid-stride loop with a few blocks per SM?
- Would `float4` loads help, and what do they require of the pointers and `n`?
