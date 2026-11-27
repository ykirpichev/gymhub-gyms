#### The idea
Both metrics are per-query averages. DCG adds a **gain** per position, `2^rel − 1` (so a grade-3 result is worth 7
times a grade-1 result), divided by a **discount** `log2(position + 1)` that shrinks slowly down the list. Dividing by
the best achievable DCG for that same query makes queries with many and few relevant results comparable. MRR looks only
at the first relevant position.

| position i | grade | gain | discount | contribution |
|---|---|---|---|---|
| 0 | 3 | 7 | 1.000 | 7.000 |
| 1 | 0 | 0 | 1.585 | 0.000 |
| 2 | 2 | 3 | 2.000 | 1.500 |

DCG = 8.5 for the first example query; its ideal order `[3, 2, 0]` scores 8.893.

#### A solution
```python
import math


def _dcg(grades, k):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def solve(rankings, k):
    ndcgs, rrs = [], []
    for grades in rankings:
        ideal = _dcg(sorted(grades, reverse=True), k)
        ndcgs.append(_dcg(grades, k) / ideal if ideal > 0 else 0.0)
        rr = 0.0
        for i, g in enumerate(grades[:k]):
            if g > 0:
                rr = 1.0 / (i + 1)
                break
        rrs.append(rr)
    return {"ndcg": sum(ndcgs) / len(ndcgs), "mrr": sum(rrs) / len(rrs)}
```

One `_dcg` helper serves twice: on the ranked grades and on the sorted grades, both cut at `k`.

#### What it costs
`O(n log n)` per query for the ideal sort (a partial sort for the top `k` is enough). Over a billion queries this is a
map-reduce: compute per-query values in parallel and average. More important than cost is what the labels mean: with
click labels, results the old ranker never showed have no grades, so offline NDCG favours rankers that resemble the
old one. Counterfactual estimators (inverse propensity weighting) and interleaving experiments address that.

#### Pitfalls
- Computing the ideal DCG from only the top `k` returned grades instead of all of the query's grades.
- Using `log2(i + 1)` with 0-based `i`, which divides by `log2(1) = 0` at the top.
- Linear gain `rel` instead of `2^rel − 1` (a valid variant, but not this spec).
- Dividing by zero for a query with no relevant result.
