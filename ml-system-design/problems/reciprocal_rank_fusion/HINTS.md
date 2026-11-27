## Hint 1

A `defaultdict(float)` from id to score; `enumerate(ranking, start=1)` gives 1-based ranks.

## Hint 2

`sorted(scores, key=lambda d: (-scores[d], d))` gives score descending, then id ascending.
