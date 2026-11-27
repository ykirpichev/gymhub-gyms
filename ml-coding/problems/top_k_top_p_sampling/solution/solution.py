import numpy as np


def solve(logits, temperature, k, p, u):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("logits must be a non-empty 1-D sequence")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be >= 0")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be >= 0")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive int")
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not np.isfinite(p) or p <= 0 or p > 1:
        raise ValueError("p must be in (0, 1]")
    if isinstance(u, bool) or not isinstance(u, (int, float)) or not np.isfinite(u) or u < 0 or u >= 1:
        raise ValueError("u must be in [0, 1)")

    vocab = int(logits.shape[0])
    if float(temperature) == 0.0:
        token = int(np.argmax(logits))
        probs = np.zeros(vocab, dtype=np.float64)
        probs[token] = 1.0
        return probs, token

    scaled = logits / float(temperature)
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    probs = exp / np.sum(exp)

    def rank(weights):
        # Higher probability first. Equal probabilities keep the lower index first.
        return np.lexsort((np.arange(weights.shape[0]), -weights))

    chosen = rank(probs)[: min(k, vocab)]
    filtered = np.zeros(vocab, dtype=np.float64)
    filtered[chosen] = probs[chosen]
    filtered /= np.sum(filtered)

    order = rank(filtered)
    order = order[filtered[order] > 0.0]
    cumulative = np.cumsum(filtered[order])
    count = int(np.searchsorted(cumulative, float(p), side="left")) + 1
    count = min(max(count, 1), int(order.shape[0]))
    kept = order[:count]
    out = np.zeros(vocab, dtype=np.float64)
    out[kept] = filtered[kept]
    out /= np.sum(out)

    cdf = np.cumsum(out)
    token = int(np.searchsorted(cdf, float(u), side="right"))
    if token == vocab:
        token = int(np.flatnonzero(out)[-1])
    return out, token
