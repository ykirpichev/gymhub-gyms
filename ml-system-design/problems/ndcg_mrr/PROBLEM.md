# NDCG and MRR for a ranker

Write `solve(rankings, k)` that evaluates a ranker.

- `rankings`: a non-empty list of queries. Each query is a list of relevance grades (ints `≥ 0`) of the results **in the order the ranker returned them**: `[3, 0, 2]` means the top result has grade 3.
- `k`: an int `≥ 1`.

For each query:

- `DCG@k = Σ (2^rel_i − 1) / log2(i + 2)` over positions `i = 0 .. min(k, len) − 1`.
- `IDCG@k` is the DCG@k of the same grades sorted from highest to lowest. `NDCG@k = DCG@k / IDCG@k`, or `0.0` when `IDCG@k = 0` (no relevant result at all).
- `RR@k = 1 / (i + 1)` for the first position `i < k` with a grade `> 0`, or `0.0` if there is none in the top `k`.

Return `{"ndcg": mean NDCG@k, "mrr": mean RR@k}` over the queries, as floats.

```
solve([[3, 0, 2], [0, 0, 1]], 3)
-> {"ndcg": 0.7279, "mrr": 0.6667}
```

Query 1: `DCG = 7/1 + 0 + 3/2 = 8.5`; ideal order `[3, 2, 0]` gives `7 + 3/log2(3) = 8.893`; NDCG `0.9558`. Query 2: `DCG = 1/2 = 0.5`, ideal `1.0`, NDCG `0.5`. RR is `1` and `1/3`.

#### What interviewers look for

- Knowing what each metric rewards: NDCG uses graded relevance and discounts by position; MRR only cares about the first relevant result.
- The ideal DCG computed from the query's own grades, and the zero case handled.
- Which metric you would report for retrieval (recall@k) against ranking (NDCG), and how offline metrics relate to online ones.

#### Follow-ups

- Your offline NDCG went up but online click-through went down. What could explain it?
- The grades come from clicks rather than human labels. What biases does that bring into this evaluation?
- How would you compute these metrics over a billion queries?
