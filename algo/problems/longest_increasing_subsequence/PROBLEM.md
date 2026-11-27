# Longest increasing subsequence

**Level:** medium   **Teaches:** dynamic programming, the "smallest tail" invariant, binary search inside a DP

## Task

A subsequence keeps some elements of `nums` in their original order (not necessarily adjacent). Return the length of the longest subsequence whose values are **strictly** increasing.

## Arguments

| Name    | Type        | Role                                   |
| ------- | ----------- | -------------------------------------- |
| `nums`  | `list[int]` | values, may be negative, may repeat; may be empty |
| returns | `int`       | the length of the longest strictly increasing subsequence |

## Example

```
nums = [10, 9, 2, 5, 3, 7, 101, 18]
one longest: 2, 3, 7, 18   (2, 5, 7, 101 is another)
result = 4
```

## How to think about it

The quadratic DP asks, for each `i`, the longest increasing subsequence ending at `i`. The faster idea keeps `tails[k]`: the smallest possible last value of an increasing subsequence of length `k + 1` seen so far. `tails` is always sorted, so each new value finds its place with a binary search and either extends the longest subsequence or lowers one tail.

## Tests

Empty input, one value, all equal, strictly increasing, strictly decreasing, duplicates inside an increasing run, negatives, and random inputs up to 3,000 values. Benchmark input: 200,000 random values. `bench` fails if the median is over **300 ms** in Python; the O(n²) DP takes hours.
