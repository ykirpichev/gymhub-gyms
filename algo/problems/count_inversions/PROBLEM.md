# Count inversions

**Level:** medium   **Teaches:** divide and conquer, counting during a merge, Fenwick trees

## Task

Count how many pairs of positions are out of order: pairs `i < j` where `nums[i] > nums[j]`. Equal values do not count.

## Arguments

| Name    | Type        | Role                              |
| ------- | ----------- | --------------------------------- |
| `nums`  | `list[int]` | values, may be negative, may repeat |
| returns | `int`       | number of inversions              |

Solve it in Python or C++. In C++ the function is:

```cpp
long long solve(vector<int>& nums);
```

The answer can reach n·(n-1)/2, about 2·10¹⁰ for the benchmark, so count in `long long`; an `int` overflows.

## Formula

```
inversions = |{ (i, j) : 0 <= i < j < n  and  nums[i] > nums[j] }|
```

## Example

```
nums = [3, 1, 2, 5, 4]
out-of-order pairs: (3,1) (3,2) (5,4)
result = 3
```

## How to think about it

Checking every pair is O(n²), far too slow for 200,000 values. Two O(n log n) routes:

- **Merge sort.** While merging two sorted halves, every time you take an element from the right half before some left-half elements, it is smaller than all the left elements still waiting. Add how many are waiting.
- **Fenwick tree.** Compress values to ranks, scan left to right, and for each value ask how many already-seen values are strictly greater.

A sorted array has 0 inversions; a reversed one of length n has n·(n-1)/2.

## Tests

Empty, single, sorted, reversed, all-equal, duplicates, and random inputs up to 3,000 values. Benchmark input: 200,000 random values. `bench` fails if the median is over **1000 ms** in Python or **40 ms** in C++; the O(n²) double loop would take hours in Python and is far over the limit in C++ too.
