# 0/1 knapsack

**Level:** medium   **Teaches:** dynamic programming over a budget, rolling arrays, iteration order

## Task

Each item `i` has a weight `weights[i]` and a value `values[i]`. Choose a set of items, each **at most once**, whose total weight is at most `capacity`, and return the largest possible total value.

## Arguments

| Name       | Type        | Role                                      |
| ---------- | ----------- | ----------------------------------------- |
| `weights`  | `list[int]` | item weights, each `>= 1`                 |
| `values`   | `list[int]` | item values, each `>= 0`, same length     |
| `capacity` | `int`       | the weight budget, `>= 0`                 |
| returns    | `int`       | the best total value (0 if nothing fits)  |

## Example

```
weights = [1, 3, 4, 5]   values = [1, 4, 5, 7]   capacity = 7
take items 1 and 2: weight 3 + 4 = 7, value 4 + 5 = 9
items 0 and 3: weight 6, value 8
result = 9
```

## How to think about it

Let `best[c]` be the most value that fits in budget `c` using the items seen so far. Adding item `i` gives each budget a new option: skip it, or take it on top of `best[c - w_i]`. Loop `c` **downwards** so that `best[c - w_i]` still means "without item `i`", which is what makes each item usable only once.

## Tests

No items, zero capacity, an item heavier than the budget, every item fitting, equal weights, a case where the greedy best-value-per-weight choice is wrong, and random inputs. Benchmark input: 200 items and `capacity = 10,000`. `bench` fails if the median is over **1000 ms** in Python; trying every subset is 2²⁰⁰ subsets.
