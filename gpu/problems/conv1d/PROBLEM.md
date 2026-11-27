# 1D convolution

**Level:** easy   **Teaches:** sliding windows, data reuse, shared and constant memory

## Task

Slide the short kernel `W` along the signal `A`. At each position, multiply the overlapping values and add them up. There is no padding, so `C` is shorter than `A` by `K - 1`.

## Arguments

| Name | Type           | Shape       | Role   |
| ---- | -------------- | ----------- | ------ |
| `A`  | `const float*` | `N`         | input  |
| `W`  | `const float*` | `K`         | input  |
| `C`  | `float*`       | `N - K + 1` | output |
| `N`  | `int`          |             | size   |
| `K`  | `int`          |             | size   |

`N >= K >= 1`.

## Formula

```
C[i] = sum over k in [0, K) of A[i + k] * W[k]    for i in [0, N - K + 1)
```

`W` is not flipped. This is what deep learning libraries call convolution (strictly, cross-correlation).

## Example

```
A = [1, 2, 3, 4]
W = [1, 0, -1]

C[0] = 1*1 + 2*0 + 3*(-1) = -2
C[1] = 2*1 + 3*0 + 4*(-1) = -2
C = [-2, -2]
```

## How to think about it

Every output is independent: it is a dot product of `W` with a window of `A`. One thread per output is a correct start.

Neighbouring windows overlap. Output `i` and output `i + 1` share `K - 1` inputs, so each element of `A` is used by `K` different threads. Every thread also reads all of `W`. In the naive version each thread loads `2K` values from global memory for `K` multiply-adds.

The benchmark uses `K = 7` and `N` around one million, which is little arithmetic per byte. The goal is to load each element of `A` from DRAM about once and serve the repeats from shared memory, registers, or cache.

## In an interview

Asked to see whether you reuse neighbouring loads through shared memory instead of reading each input K times.

Follow-ups to answer out loud once your kernel passes:

- How big is the halo for a kernel of width K, and who loads it?
- Put the filter in constant memory. Why does that help when every thread reads the same weight?
- How does the arithmetic intensity change as K grows?
