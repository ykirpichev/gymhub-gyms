# Hints for Top k frequent words

## Hint 1

`collections.Counter(words)` gives you each word's count in one pass.

## Hint 2

You need one sort key that puts high counts first and, among equal counts, small words first. The tuple `(-count, word)` does both, because tuples compare element by element.

## Hint 3

`sorted(counts, key=...)[:k]` works. `heapq.nsmallest(k, counts, key=...)` gives the same answer without sorting every distinct word. Watch out for `Counter.most_common(k)`: its tie order follows first appearance, not the alphabet.

In C++, count with `unordered_map<string, int>`, copy the entries into a vector, and use `partial_sort` (or `nth_element` then `sort`) with a comparator on `(count descending, word ascending)` to order only the first `k`.
