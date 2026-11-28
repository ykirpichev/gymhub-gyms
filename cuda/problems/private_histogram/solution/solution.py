def solve(values, bins, block):
    hist = [0] * bins
    shared = glob = 0
    for start in range(0, len(values), block):
        private = [0] * bins
        for v in values[start:start + block]:
            private[v] += 1
            shared += 1
        for b in range(bins):
            if private[b]:
                hist[b] += private[b]
                glob += 1
    return [hist, shared, glob]
