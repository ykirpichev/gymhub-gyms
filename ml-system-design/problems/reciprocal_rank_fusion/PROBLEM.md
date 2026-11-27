# Reciprocal rank fusion

Write `solve(rankings, k, top_n)` that merges the results of several retrievers.

- `rankings`: a list of ranked lists of document ids (strings), each best first. A list never repeats an id; different lists may share ids. Any list may be empty.
- `k`: a positive number, the RRF constant (60 is the usual choice).
- `top_n`: an int `≥ 1`.

Each document's score is the sum, over the lists that contain it, of `1 / (k + rank)`, where `rank` is its **1-based** position in that list. Return the `top_n` highest-scoring ids (all of them if there are fewer), best first. Break ties by id in ascending string order.

```
solve([["a", "b", "c"], ["b", "c", "d"]], 60, 4)  ->  ["b", "c", "a", "d"]
```

`b` scores `1/62 + 1/61 = 0.03252`, `c` scores `1/63 + 1/62 = 0.03200`, `a` scores `1/61` and `d` scores `1/63`.

#### What interviewers look for

- Why rank fusion: BM25 scores and cosine similarities are on different scales, so adding raw scores needs calibration; ranks do not.
- What `k` does: a larger `k` flattens the difference between rank 1 and rank 10, so agreement across lists matters more than one list's top spot.
- Where fusion sits in the pipeline: after retrieval, before the more expensive cross-encoder re-ranker.

#### Follow-ups

- How would you weight one retriever more than the other?
- What would you do instead if both retrievers returned calibrated probabilities?
- Your keyword retriever returns 1,000 results and the vector one 50. Does that bias RRF?
