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
