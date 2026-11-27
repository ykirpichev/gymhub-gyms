# Top k frequent words

**Level:** easy   **Teaches:** counting with a hash map, sorting by a compound key, heaps

## Task

Return the `k` words that appear most often in `words`, most frequent first. When two words appear the same number of times, the one that comes first in plain string order (`"a" < "ab" < "b"`, uppercase before lowercase) goes first.

## Arguments

| Name    | Type        | Role                                                       |
| ------- | ----------- | ---------------------------------------------------------- |
| `words` | `list[str]` | the words, with repeats                                    |
| `k`     | `int`       | how many to return, `1 <= k <=` number of distinct words   |
| returns | `list[str]` | `k` distinct words in rank order                           |

Solve it in Python or C++. In C++ the function is:

```cpp
vector<string> solve(vector<string>& words, int k);
```

`std::string`'s `<` orders strings the same way as Python's for these words.

## Rules

- Rank by count, highest first.
- Break ties with Python's default string comparison, smallest first.
- Each word appears at most once in the output.

## Example

```
words = ["pear", "fig", "pear", "kiwi", "fig", "pear", "apple"]   k = 2
counts: pear 3, fig 2, apple 1, kiwi 1
result = ["pear", "fig"]
```

## How to think about it

Count once, then order the distinct words by the pair `(-count, word)`. Sorting all distinct words is O(u log u) for `u` distinct words. A heap of size `k` (`heapq.nsmallest`) gets that down to O(u log k), which matters when `k` is much smaller than `u`.

## Tests

Ties, `k` equal to the number of distinct words, a single repeated word, all-distinct words, prefixes, mixed case, and random inputs. Benchmark input: 500,000 words over about 8,000 distinct words, `k = 100`. There is no time limit in either language.
