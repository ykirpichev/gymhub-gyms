#### The idea
Different retrievers produce scores on incomparable scales (BM25 is unbounded, cosine similarity lies in `[−1, 1]`, and
both shift from query to query), but their **ranks** are comparable. RRF gives each appearance a vote of
`1 / (k + rank)` and sums the votes: documents near the top of several lists win.

| doc | keyword rank | vector rank | score with k = 60 |
|---|---|---|---|
| b | 2 | 1 | 1/62 + 1/61 = 0.0325 |
| c | 3 | 2 | 1/63 + 1/62 = 0.0320 |
| a | 1 | none | 1/61 = 0.0164 |
| d | none | 3 | 1/63 = 0.0159 |

#### A solution
```python
from collections import defaultdict


def solve(rankings, k, top_n):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1.0 / (k + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))[:top_n]
```

#### What it costs
`O(total list length + D log D)` for `D` distinct ids. The cost is irrelevant next to the retrievers and the LLM; RRF's
appeal is that it is robust and has a single parameter. With `k = 60`, the difference between rank 1 and rank 2 is tiny
(`1/61` against `1/62`), so a document both retrievers like beats one that only a single retriever ranks first. To
weight retrievers, multiply each list's votes by a weight. If you have relevance labels, a learned combination or a
cross-encoder re-ranker on the fused list does better, and production systems usually do both: RRF to merge, then a
re-ranker.

#### Pitfalls
- 0-based ranks, which give the top document `1/k` and change every score.
- Summing raw retriever scores instead of rank votes.
- Breaking ties by insertion order instead of the stated id order.
- Forgetting that a document missing from a list simply gets no vote from it (not a penalty).
