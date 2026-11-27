# Top-k per row

**Level:** hard   **Teaches:** selection without a full sort, per-thread heaps, warp-level merging, radix select

## Task

Top-k sampling keeps only the `k` most likely tokens before drawing one. For each row of `X`, write its `k` largest values to the same row of `Y`, largest first. Values only, no indices. This is `torch.topk(X, k, dim=-1).values`.

## Arguments

| Name   | Type           | Shape         | Role   |
| ------ | -------------- | ------------- | ------ |
| `X`    | `const float*` | `rows × cols` | input  |
| `Y`    | `float*`       | `rows × k`    | output |
| `rows` | `int`          |               | size   |
| `cols` | `int`          |               | size   |
| `k`    | `int`          |               | size   |

`1 <= k <= cols` and `k <= 64`. Rows are long, up to about 50,000 columns (a vocabulary). Values can repeat; a repeated value appears in `Y` as many times as it is among the top `k`. The comparison is exact.

## Formula

```
Y[r, :] = the k largest entries of X[r, :], in decreasing order
```

## Example

```
X = [[3, 1, 4, 1, 5],      k = 2
     [9, 2, 6, 5, 3]]

Y = [[5, 4],
     [9, 6]]
```

## How to think about it

Sorting each row and taking the first `k` is correct, but it does far more work than needed: a 32,768-long row needs only 50 values. Selection keeps a small candidate set instead.

One way: give each row a block. Each thread scans its share of the row and keeps its own top `k` in registers or shared memory, a sorted list where a new value that beats the smallest one is inserted. Then merge the threads' lists, pairwise in a tree, until one list of `k` remains. The scan reads the row once, which is all the memory traffic there is, and the merge is small.

Another way is radix select: look at the bits of the values from the top, count how many values fall in each bucket with a histogram, and narrow down to the bucket that holds the `k`-th largest value. Then collect everything above it. Floats can be compared as integers after flipping their bits so that the order is preserved: `u = bits ^ ((bits >> 31) ? 0xFFFFFFFF : 0x80000000)`.

## In an interview

Selection on a GPU: sorting everything is wasteful when k is small.

Follow-ups to answer out loud once your kernel passes:

- What would you do for k = 1, for k = 50 of a 150K vocabulary, and for k = n / 2?
- How does top-p sampling relate, and why does it need a sort or a scan?
- How do you break ties deterministically?
