def solve(buffers):
    n = len(buffers)
    if n == 0:
        raise ValueError("need at least one rank")
    length = len(buffers[0])
    if any(len(b) != length for b in buffers) or length % n:
        raise ValueError("buffers must share a length divisible by the rank count")
    size = length // n
    data = [list(b) for b in buffers]

    def chunk(r, c):
        return data[r][c * size:(c + 1) * size]

    sent = 0
    for s in range(n - 1):
        messages = [((r + 1) % n, (r - s) % n, chunk(r, (r - s) % n)) for r in range(n)]
        for dest, c, values in messages:
            for k, v in enumerate(values):
                data[dest][c * size + k] += v
        sent += size
    after = [list(b) for b in data]
    for s in range(n - 1):
        messages = [((r + 1) % n, (r + 1 - s) % n, chunk(r, (r + 1 - s) % n)) for r in range(n)]
        for dest, c, values in messages:
            data[dest][c * size:(c + 1) * size] = values
        sent += size
    return (after, data, sent)
