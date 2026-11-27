# SAXPY

**Level:** easy   **Teaches:** scalar arguments, fusing operations into one pass

## Task

Scale `A` by the number `alpha`, add `B`, and write the result to `C`. The name comes from BLAS: Single-precision A times X Plus Y.

## Arguments

| Name    | Type           | Shape | Role   |
| ------- | -------------- | ----- | ------ |
| `A`     | `const float*` | `N`   | input  |
| `B`     | `const float*` | `N`   | input  |
| `C`     | `float*`       | `N`   | output |
| `alpha` | `float`        |       | scalar |
| `N`     | `int`          |       | size   |

`alpha` is passed by value, not as a pointer.

## Formula

```
C[i] = alpha * A[i] + B[i]    for i in [0, N)
```

## Example

```
alpha = 2
A = [1, 2]
B = [3, 4]
C = [2*1 + 3, 2*2 + 4] = [5, 8]
```

## How to think about it

Like vector add, every element is independent, so one thread per element works. `alpha` is the same for every thread. It arrives as a kernel argument and needs no memory load per element.

The kernel is memory-bound: 12 bytes per element for one fused multiply-add. The main lesson is to do it in one pass. Computing `alpha * A` into a temporary and then adding `B` reads and writes memory twice, which is roughly 1.7 times the traffic.

## In an interview

The classic bandwidth test. Interviewers expect you to count bytes: two reads and one write per element.

Follow-ups to answer out loud once your kernel passes:

- Your SAXPY reaches 60% of the copy bandwidth. What would you check first?
- Why does an in-place write to `y` still cost a full read and a full write?
- How would you overlap a host-to-device copy of the next chunk with this kernel? (Streams and pinned memory.)
