import math


def solve(arrivals, max_batch, max_wait, base_ms, per_item_ms):
    n = len(arrivals)
    latency = [0] * n
    i, free = 0, 0
    while i < n:
        oldest = arrivals[i]
        full = arrivals[i + max_batch - 1] if i + max_batch - 1 < n else math.inf
        start = max(free, oldest, min(full, oldest + max_wait))
        j = i
        while j < n and j - i < max_batch and arrivals[j] <= start:
            j += 1
        done = start + base_ms + per_item_ms * (j - i)
        for r in range(i, j):
            latency[r] = done - arrivals[r]
        free, i = done, j
    return latency
