# Sort

**Level:** hard   **Teaches:** bitonic sorting networks, shared-memory sorts, radix sort from histograms and scans

## Task

Write the values of `A` to `B` in ascending order. This is `torch.sort(A).values`.

## Arguments

| Name | Type           | Shape | Role   |
| ---- | -------------- | ----- | ------ |
| `A`  | `const float*` | `N`   | input  |
| `B`  | `float*`       | `N`   | output |
| `N`  | `int`          |       | size   |

`N` is any size up to 16 million and need not be a power of two. Values can repeat. There are no `NaN`s.

## Formula

```
B = A sorted so that B[0] <= B[1] <= ... <= B[N - 1]
```

## Example

```
A = [3, -1, 2.5, 0, -7, 2.5]
B = [-7, -1, 0, 2.5, 2.5, 3]
```

## How to think about it

A GPU sort cannot be a quicksort: the work must be split into many independent, equal pieces. Two families do that.

A **bitonic sort** is a fixed network of compare-and-swap steps. For a power-of-two size `n`, it runs `log2(n) * (log2(n) + 1) / 2` stages, and in every stage each of `n / 2` threads compares two elements at a known distance and swaps them if they are in the wrong order. No step depends on the data, so it maps directly to threads. Sizes that are not a power of two can be padded with `+inf`. Stages whose distance is small fit inside one block and can run in shared memory with `__syncthreads()` between them, which saves most of the global memory traffic.

A **radix sort** processes the bits of the keys in groups of, say, 8. For each group it counts how many keys fall in each of the 256 buckets (`histogram`), turns the counts into starting offsets (`prefix_sum`), and scatters every key to its bucket's next slot, keeping the order of equal digits. Four passes sort 32-bit keys. Floats sort correctly as unsigned integers after one bit trick: flip all bits of negative numbers and only the sign bit of positive ones.

Bitonic is the simpler one to get right. Radix sort reads and writes the array a fixed number of times and is what libraries use for large `N`.

## In an interview

A classic parallel algorithm question: bitonic sort or radix sort, and their costs.

Follow-ups to answer out loud once your kernel passes:

- Bitonic sort does O(n log² n) comparisons. Why use it on a GPU anyway?
- Explain one pass of a radix sort: a histogram, a scan, a scatter.
- Where does sorting appear in model code? (MoE token grouping, top-k and top-p, deduplication.)
