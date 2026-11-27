# Hints for BPE training

## Hint 1

Work on distinct words: `collections.Counter(text.split())`. Store each word as a list of symbols and its frequency. The tie-break is exactly Python's built-in ordering: `min(candidates, key=lambda p: (-count[p], p))`.

## Hint 2

Keep `count[pair]` and `where[pair] = set of word ids` across merges. When merging `(a, b)`, visit only `where[(a, b)]`. For each such word, subtract `freq` from every pair it currently has, rebuild the symbol list, then add `freq` to every pair of the new list and update `where`. Iterate over a copy of the set, since you modify it.

## Hint 3

Replace the per-step scan with `heapq`. Push `(-count, pair)` whenever a pair's count changes. When you pop, compare against the live `count[pair]`; if they differ the entry is stale, so drop it and pop again. Python compares the tuples by count first, then by `(a, b)`, which is the required tie-break.
