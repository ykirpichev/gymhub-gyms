#### The idea
Follow the schedule literally, with one rule that is easy to miss: in each step every rank sends *simultaneously*,
so collect all messages from the state before the step, then apply them.

#### A solution
```python
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
    for s in range(n - 1):                       # reduce-scatter: add
        messages = [((r + 1) % n, (r - s) % n, chunk(r, (r - s) % n)) for r in range(n)]
        for dest, c, values in messages:
            for k, v in enumerate(values):
                data[dest][c * size + k] += v
        sent += size
    after = [list(b) for b in data]
    for s in range(n - 1):                       # all-gather: overwrite
        messages = [((r + 1) % n, (r + 1 - s) % n, chunk(r, (r + 1 - s) % n)) for r in range(n)]
        for dest, c, values in messages:
            data[dest][c * size:(c + 1) * size] = values
        sent += size
    return (after, data, sent)
```

#### The cost
Each rank sends `2(N − 1)` chunks of `L / N`: `2(N − 1)/N · L` elements, under `2L` however many ranks there are.
Every link carries one chunk per step, so the time is about `2L·bytes / link_bandwidth` plus `2(N − 1)` latencies.
The latency term is why large clusters use trees or hierarchical rings (inside a node over NVLink, then across nodes)
for small messages.

#### Pitfalls
- Applying messages as they are built: a rank forwards a chunk it already received in the same step.
- Python's `%` is never negative, so `(r - s) % n` is safe; in C it would need `+ n`.
