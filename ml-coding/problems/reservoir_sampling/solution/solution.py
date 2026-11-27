import random


def solve(stream, k, seed):
    if k < 0:
        raise ValueError("k must be non-negative")
    rng = random.Random(seed)
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            # Keep item i with probability k / (i + 1); it replaces a uniformly chosen slot.
            j = rng.randrange(i + 1)
            if j < k:
                reservoir[j] = item
    return reservoir
