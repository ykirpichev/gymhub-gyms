#### The idea
Count with a hash map, then rank the distinct words by the key `(-count, word)`: higher count first, and on equal
counts the smaller string first. `heapq.nsmallest(k, ...)` with that key keeps only `k` candidates at a time.

#### A solution
```python
import heapq
from collections import Counter


def solve(words, k):
    counts = Counter(words)
    return heapq.nsmallest(k, counts, key=lambda w: (-counts[w], w))
```

#### What it costs
Counting is `O(n)`. Ranking `u` distinct words is `O(u log k)` with the heap, `O(u log u)` with a full sort, and the
difference only shows when `k ≪ u`. For a stream too large for memory, a count-min sketch plus a heap gives
approximate heavy hitters in fixed space. In C++, `unordered_map<string, int>` and `std::partial_sort` with a comparator
on `(count desc, word asc)` are the direct translation.

#### Pitfalls
- `Counter.most_common(k)`: it breaks ties by first appearance, not by string order.
- A heap of `(count, word)` pairs as a max-heap by negation, popping `k` times, and forgetting that the tie-break must
  then be on the word **ascending**.
- Returning `(word, count)` pairs instead of words.
