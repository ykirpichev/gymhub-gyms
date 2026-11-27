# nDCG at 3

Use the graded discounted formula. For a ranking with relevance `rel_i` at position `i` (positions start at 1),

`DCG = Σ_i (2^rel_i − 1) / log2(i + 1)`

`nDCG@k = DCG@k / IDCG@k`, where IDCG is the DCG of the ideal order.

The system returns relevances 1, then 0, then 3. The ideal order of these grades is 3, 1, 0. Compute nDCG@3.

Give the decimal to at least three places.
