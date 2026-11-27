import math
import random
from collections import Counter, defaultdict


_MODES = ("uniform_with_replacement", "uniform_without_replacement", "weighted", "stratified")


def solve(n, k, mode, seed, weights, labels):
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    if mode not in _MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if k > 0 and n == 0:
        raise ValueError("cannot draw from an empty population")
    if mode in ("uniform_without_replacement", "stratified") and k > n:
        raise ValueError("k exceeds n for a sample without replacement")
    if mode == "weighted":
        _check_weights(n, k, weights)
    if mode == "stratified" and (not isinstance(labels, (list, tuple)) or len(labels) != n):
        raise ValueError("labels must have one label per index")
    if k == 0:
        return []
    rng = random.Random(seed)
    if mode == "uniform_with_replacement":
        return [rng.randrange(n) for _ in range(k)]
    if mode == "uniform_without_replacement":
        return rng.sample(range(n), k)
    if mode == "weighted":
        return [_draw_weighted(rng, weights) for _ in range(k)]
    return _stratified(n, k, rng, labels)


def _check_weights(n, k, weights):
    if not isinstance(weights, (list, tuple)) or len(weights) != n:
        raise ValueError("weights must have one number per index")
    total = 0.0
    for weight in weights:
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError("weights must be non-negative")
        total += float(weight)
    if k > 0 and total <= 0:
        raise ValueError("weights must sum to a positive number")


def _draw_weighted(rng, weights):
    total = math.fsum(float(weight) for weight in weights)
    ticket = rng.random() * total
    running = 0.0
    last = 0
    for index, weight in enumerate(weights):
        if weight <= 0:
            continue
        last = index
        running += float(weight)
        if ticket < running:
            return index
    return last


def _stratified(n, k, rng, labels):
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    quotas = _quotas(labels, k)
    picked = []
    for label in sorted(groups):
        items = groups[label][:]
        rng.shuffle(items)
        picked.extend(items[:quotas[label]])
    return picked


def _quotas(labels, k):
    n = len(labels)
    counts = Counter(labels)
    order = sorted(counts)
    floors = [(k * counts[label]) // n for label in order]
    remainders = [(k * counts[label]) % n for label in order]
    left = k - sum(floors)
    rank = sorted(range(len(order)), key=lambda i: (-remainders[i], str(order[i])))
    for i in rank[:left]:
        floors[i] += 1
    return {order[i]: floors[i] for i in range(len(order))}
