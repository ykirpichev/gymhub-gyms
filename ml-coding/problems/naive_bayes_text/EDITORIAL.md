#### The idea
Training is counting. For each class, count documents (the prior) and token occurrences (the likelihood), then add
`alpha` to every count so a token never seen in a class does not zero out the whole product (Laplace smoothing).
Scoring multiplies many small probabilities, so do it in log space: a sum of logs never underflows.

Precompute `log P(t | c)` for the whole vocabulary once. Scoring a document is then a dictionary lookup per token.

#### A solution
```python
import math
from collections import Counter


def solve(train_docs, train_labels, test_docs, alpha):
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    classes = sorted(set(train_labels))
    vocab = {token for doc in train_docs for token in doc}
    counts = {c: Counter() for c in classes}
    n_docs = Counter(train_labels)
    for doc, label in zip(train_docs, train_labels):
        counts[label].update(doc)

    log_prior = {c: math.log(n_docs[c] / len(train_labels)) for c in classes}
    log_lik = {}
    for c in classes:
        denom = sum(counts[c].values()) + alpha * len(vocab)
        log_lik[c] = {t: math.log((counts[c][t] + alpha) / denom) for t in vocab}

    predictions, scores = [], []
    for doc in test_docs:
        row = [log_prior[c] + sum(log_lik[c][t] for t in doc if t in vocab) for c in classes]
        top = max(row)
        best = next(k for k, score in enumerate(row) if score >= top - 1e-9)  # ties: first class in sorted order
        predictions.append(classes[best])
        scores.append(row)
    return predictions, scores
```

#### What it costs
Training is `O(total tokens + K·|V|)`. Scoring a document of `L` tokens is `O(K·L)`. In a vectorised version, the
documents become a sparse count matrix `M × |V|` and the scores are one sparse matmul with the `|V| × K` log-likelihood
matrix plus the log prior.

#### Pitfalls
- Multiplying probabilities instead of adding logs: a document of a few hundred tokens underflows to 0 for every class.
- Using `|V|` per class instead of the global vocabulary in the denominator.
- Counting each token once per document (that is Bernoulli Naive Bayes, a different model).
- Letting out-of-vocabulary tokens raise a `KeyError` or add a smoothed term.
- Breaking ties by dictionary order instead of sorted class order.
