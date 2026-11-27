# Histogram

**Level:** medium   **Teaches:** atomic operations, contention, per-block shared-memory histograms

## Task

`A` holds `N` whole numbers stored as floats, each in `[0, bins)`. Count how many times each value appears. Write the count for value `b` to `counts[b]`.

`counts` starts out filled with NaN, not zeros. Set it to zero yourself before you start adding, for example with `cudaMemset(counts, 0, bins * sizeof(float))` or a small kernel that writes `0.0f`.

## Arguments

| Name     | Type           | Shape  | Role   |
| -------- | -------------- | ------ | ------ |
| `A`      | `const float*` | `N`    | input  |
| `counts` | `float*`       | `bins` | output |
| `N`      | `int`          |        | size   |
| `bins`   | `int`          |        | size   |

`bins` is at most 1024, so a whole histogram fits in shared memory. Counts must be exact: the test uses zero tolerance. Every count stays below 2^24, so float32 holds it exactly.

## Formula

```
counts[b] = number of i in [0, N) with A[i] == b
```

Read the bin with `int b = (int)A[i];`.

## Example

```
A    = [0, 1, 1, 3, 1, 0]    bins = 4
counts = [2, 3, 0, 1]
```

Value 2 never appears, so `counts[2]` is 0. It must still be written.

## How to think about it

One thread per element is the natural split. The problem is the write: many threads want to add 1 to the same bin at the same time. A plain `counts[b] += 1` loses updates, because two threads can read the old value before either writes. `atomicAdd(&counts[b], 1.0f)` makes each add indivisible, and it is correct.

It is also slow when many threads hit the same few bins, because the adds to one address happen one after another. The fix is to keep a private histogram per block in shared memory, where atomics are cheap, and then add each block's histogram to the global one once at the end.

## In an interview

Tests atomics and contention: what happens when many threads add to the same bin.

Follow-ups to answer out loud once your kernel passes:

- Why is a histogram of mostly-equal values much slower than one of uniform values?
- Privatize the histogram per block in shared memory. When does it stop fitting?
- Are floating-point atomic sums deterministic? What about integer counts?
