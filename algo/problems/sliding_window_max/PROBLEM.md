# Sliding window maximum

**Level:** medium   **Teaches:** monotonic deque, amortized O(1) per element

## Task

Slide a window of width `k` across `nums` one step at a time, from the left edge to the right edge. Return the largest value inside the window at every position.

## Arguments

| Name    | Type        | Role                                              |
| ------- | ----------- | ------------------------------------------------- |
| `nums`  | `list[int]` | values, may be negative, may repeat               |
| `k`     | `int`       | window width, `1 <= k`, and `k <= len(nums)` when `nums` is non-empty |
| returns | `list[int]` | `len(nums) - k + 1` maxima (empty if `nums` is empty) |

Solve it in Python or C++. In C++ the function is:

```cpp
vector<int> solve(vector<int>& nums, int k);
```

## Formula

```
out[i] = max(nums[i], nums[i+1], ..., nums[i+k-1])     for i = 0 .. len(nums) - k
```

## Example

```
nums = [1, 3, -1, -3, 5, 3, 6, 7]   k = 3
windows: [1 3 -1] [3 -1 -3] [-1 -3 5] [-3 5 3] [5 3 6] [3 6 7]
out    = [3, 3, 5, 5, 6, 7]
```

## How to think about it

Recomputing `max` of every window costs O(n·k). Instead, keep the indices of the elements that could still become a window maximum. An element is useless once a larger (or equal) element arrives after it, because the newer one stays in the window longer. Those survivors form a decreasing sequence, so the current maximum is always at the front. Each index is pushed and popped at most once, so the whole pass is O(n).

## Tests

Empty input, `k = 1`, `k = len(nums)`, strictly increasing and decreasing runs, all-equal values, and random inputs with many duplicates. Benchmark input: `n = 200,000`, `k = 5,000`. `bench` fails if the median is over **100 ms** in Python or **6 ms** in C++; an O(n·k) solution takes many seconds in Python and is far over the limit in C++ too.
