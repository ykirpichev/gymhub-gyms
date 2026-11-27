#### The idea
Build a token-to-column map from the sorted vocabulary, fill a count matrix row by row with `Counter`, then everything
else is vectorised: document frequency is the number of non-zero entries per column, IDF is a formula over it, and each
row is scaled to unit length.

TF says "this word is frequent here"; IDF says "and rare elsewhere". A word in every document gets the minimum IDF of
1, so it still counts a little (that is what the `+ 1` after the log does).

#### A solution
```python
from collections import Counter

import numpy as np


def solve(docs):
    vocab = sorted({token for doc in docs for token in doc})
    column = {token: j for j, token in enumerate(vocab)}
    n = len(docs)
    tf = np.zeros((n, len(vocab)))
    for i, doc in enumerate(docs):
        for token, count in Counter(doc).items():
            tf[i, column[token]] = count
    df = np.count_nonzero(tf, axis=0)
    idf = np.log((1 + n) / (1 + df)) + 1
    tfidf = tf * idf
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vocab, tfidf / norms
```

Setting zero norms to 1 leaves empty documents as rows of zeros without a division warning.

#### What it costs
This dense matrix is `O(N·V)` memory, which is fine for a toy corpus and impossible for a real one (a million
documents by a million tokens). Real implementations use a sparse CSR matrix, `O(total tokens)`. Scoring a query
against the corpus is then a sparse matrix-vector product; BM25 replaces raw TF with a saturating function and adds
document-length normalisation.

#### Pitfalls
- Document frequency counted as total occurrences instead of the number of documents.
- `log(N / df)` without smoothing: a word in every document gets weight 0 and an unseen one divides by zero.
- Normalising columns instead of rows.
- Dividing an empty document by a zero norm, producing `nan`.
- Building the vocabulary in first-seen order instead of sorted order.
