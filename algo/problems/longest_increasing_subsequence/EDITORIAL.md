#### The idea
Keep `tails[k]` = the smallest last value of any strictly increasing subsequence of length `k + 1` found so far. A
smaller tail is always better, since it leaves more room to extend. `tails` is strictly increasing (a subsequence of
length `k + 2` contains one of length `k + 1` with a smaller tail), so a new value `x`:

- extends the longest subsequence if it is larger than every tail (append);
- otherwise replaces the first tail `≥ x`, making that length's tail smaller.

Walking `[10, 9, 2, 5, 3, 7, 101, 18]`:

| x | tails after |
|---|---|
| 10 | [10] |
| 9 | [9] |
| 2 | [2] |
| 5 | [2, 5] |
| 3 | [2, 3] |
| 7 | [2, 3, 7] |
| 101 | [2, 3, 7, 101] |
| 18 | [2, 3, 7, 18] |

`tails` is not itself a subsequence of the input in general; only its length is the answer.

#### A solution
```python
import bisect


def solve(nums):
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)   # first tail >= x: equal values do not extend
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)
```

#### What it costs
`O(n log n)` time and `O(L)` memory, where `L` is the answer. The textbook DP is `O(n²)`: 4·10¹⁰ steps for the
benchmark. To recover the subsequence itself, store for each element the index of its predecessor (the element at
`tails[i − 1]` when it was placed) and walk back from the last tail. For non-decreasing subsequences, use
`bisect_right` instead. The same structure computes the longest chain of nested envelopes (sort by width, then LIS on
height) and the minimum number of piles in patience sorting.

#### Pitfalls
- `bisect_right` for a strictly increasing subsequence: equal values then extend the run.
- Returning `tails` as if it were the subsequence.
- Confusing subsequence with subarray (contiguous), which is a simple linear scan.
