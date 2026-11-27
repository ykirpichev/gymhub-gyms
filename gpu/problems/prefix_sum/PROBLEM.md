# Prefix sum

**Level:** medium   **Teaches:** scans inside a block, multi-block scans with block sums

## Task

Write the running total of `A` into `B`. `B[i]` is the sum of `A[0]` through `A[i]`, including `A[i]` itself. This is called an inclusive scan.

## Arguments

| Name | Type           | Shape | Role   |
| ---- | -------------- | ----- | ------ |
| `A`  | `const float*` | `N`   | input  |
| `B`  | `float*`       | `N`   | output |
| `N`  | `int`          |       | size   |

## Formula

```
B[i] = A[0] + A[1] + ... + A[i]    for i = 0 .. N-1
```

The inputs are in `[-1, 1]`. Float32 rounding grows with `N`, so the check allows a small absolute error. Exclusive scans (where `B[i]` leaves out `A[i]`) fail.

## Example

```
A = [3, 1, 4, 1,  5]
B = [3, 4, 8, 9, 14]
```

## How to think about it

A loop is trivial on one thread: `B[i] = B[i-1] + A[i]`. But each step waits on the one before it, so it cannot use the rest of the GPU.

The trick is that addition is associative, so you can group the sums in any order. Inside one block, threads can build all running totals in about `log2(blockDim)` rounds. At round `d`, every element adds the value `2^d` places to its left. Blocks cannot see each other's shared memory, so a long array needs three steps:

1. Each block scans its own chunk and records the chunk's total.
2. Scan the list of chunk totals.
3. Each block adds the total of all chunks before it to its own values.

## In an interview

Scan is a favourite because it has a naive O(n log n) version and a work-efficient one, and it must work across blocks.

Follow-ups to answer out loud once your kernel passes:

- Explain the up-sweep and down-sweep, and why the work-efficient scan does O(n) additions.
- How do you scan an array larger than one block? (Scan blocks, scan the block sums, add back.)
- Name two algorithms built on scan. (Stream compaction, radix sort, top-p sampling.)
