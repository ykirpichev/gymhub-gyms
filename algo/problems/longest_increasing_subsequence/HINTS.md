# Hints for Longest increasing subsequence

## Hint 1

The O(n²) version: `best[i] = 1 + max(best[j] for j < i if nums[j] < nums[i])`. It is correct and too slow for the benchmark.

## Hint 2

Keep a list `tails`, where `tails[k]` is the smallest tail of any increasing subsequence of length `k + 1`. Why is `tails` sorted?

## Hint 3

For each `x`, find `i = bisect_left(tails, x)`. If `i == len(tails)`, append `x`; otherwise set `tails[i] = x`. `bisect_left` (not `bisect_right`) keeps the subsequence strictly increasing. The answer is `len(tails)`.
