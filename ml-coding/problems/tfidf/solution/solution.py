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
