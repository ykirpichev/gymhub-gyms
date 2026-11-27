# Exact cosine top-k retrieval

Write `solve(queries, docs, k)`: the retrieval step of a RAG pipeline, done exactly (no approximate index).

- `queries`: `Q × d` embeddings (list of lists or array). `Q ≥ 1`, `d ≥ 1`.
- `docs`: `N × d` embeddings. `N ≥ 1`.
- `k`: an int `≥ 1`.

The score of document `j` for query `i` is the cosine similarity `qᵢ · dⱼ / (‖qᵢ‖ ‖dⱼ‖)`. If either vector is all zeros, the score is `0.0`.

Return a tuple `(indices, scores)`, both `Q × m` arrays with `m = min(k, N)`:

- `indices`: integer array; row `i` lists the document indices of the `m` best documents for query `i`, by score **descending**. Equal scores are ordered by **lower document index first**.
- `scores`: float array; `scores[i, t]` is the score of document `indices[i, t]`.

If `k > N`, return all `N` documents in that order. Raise `ValueError` if `k < 1`, or if `queries` and `docs` are not 2-D with the same `d`.

Hidden tests include thousands of documents; score all of them with matrix operations rather than a Python loop over pairs.

```
solve(queries=[[1, 0], [0, 0]], docs=[[0, 2], [3, 3], [5, 0], [1, 0]], k=3)
-> (array([[2, 3, 1], [0, 1, 2]]),
    array([[1.0, 1.0, 0.7071], [0.0, 0.0, 0.0]]))
```

Documents 2 and 3 tie for the first query, so 2 comes first. The second query is all zeros, so every score is 0 and the order is by index.

**Interview focus:** normalizing both sides, top-k ordering with deterministic ties, zero vectors, and how an approximate index (IVF, HNSW) would trade exactness for speed at scale.

**Follow-ups** to answer out loud once your solution passes:

- How would you chunk the documents, and what overlap would you use?
- When does hybrid search (keyword plus vector) beat vector search alone?
- How would you measure retrieval quality without labels?
