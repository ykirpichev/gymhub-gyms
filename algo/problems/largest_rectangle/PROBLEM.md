# Largest rectangle in a histogram

**Level:** hard   **Teaches:** monotonic stacks, "nearest smaller element" on both sides

## Task

`heights` describes a bar chart: bar `i` has width 1 and height `heights[i]`, and the bars stand side by side. Return the area of the largest rectangle that fits entirely under the bars. The rectangle's sides are parallel to the axes and its bottom sits on the baseline.

## Arguments

| Name      | Type        | Role                                            |
| --------- | ----------- | ----------------------------------------------- |
| `heights` | `list[int]` | bar heights, each `>= 0`; may be empty          |
| returns   | `int`       | the largest area (0 for no bars)                |

## Formula

```
area = max over 0 <= l <= r < n of  (r - l + 1) * min(heights[l..r])
```

## Example

```
heights = [2, 1, 5, 6, 2, 3]
bars 2 and 3 (heights 5 and 6) give width 2 * height 5 = 10
bars 2..5 give width 4 * height 2 = 8
result = 10
```

## How to think about it

The best rectangle has the height of some bar `i`, and stretches left and right until the first bar that is lower. So for every bar you want the nearest lower bar on each side. A stack of indices with increasing heights finds both in one pass: when a lower bar arrives, every taller bar it pops has just found its right boundary, and the bar below it on the stack is its left boundary.

## Tests

Empty input, one bar, all equal, strictly increasing, strictly decreasing, zeros in the middle, a valley, and random inputs up to 3,000 bars. Benchmark input: 200,000 random bars. `bench` fails if the median is over **200 ms** in Python; trying every pair `(l, r)` is O(n²) and takes hours.
