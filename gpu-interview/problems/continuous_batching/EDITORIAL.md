#### The idea
Simulate step by step. At the start of each step, admit from the waiting queue (sorted by arrival, then index)
according to the mode; every running request then makes one token; those that reach their length finish at `t + 1`.
When nothing runs and the next arrival is in the future, jump the clock to it.

#### A solution
```python
def solve(requests, max_batch, mode):
    if mode not in ("continuous", "static") or max_batch < 1:
        raise ValueError("bad mode or batch size")
    waiting = sorted(range(len(requests)), key=lambda i: (requests[i][0], i))
    running, finish, t = {}, [None] * len(requests), 0
    while waiting or running:
        if not running and waiting and requests[waiting[0]][0] > t:
            t = requests[waiting[0]][0]                       # idle until the next arrival
        if mode == "continuous" or not running:
            while waiting and len(running) < max_batch and requests[waiting[0]][0] <= t:
                running[waiting.pop(0)] = 0
        for index in list(running):
            running[index] += 1
            if running[index] == requests[index][1]:
                finish[index] = t + 1
                del running[index]
        t += 1
    return finish
```

#### What it shows
With static batching, a batch runs until its *longest* request ends, so a short reply that arrives one step late can
wait for a long one to finish. Continuous batching fills each freed slot on the next step, so throughput stays near
`max_batch` tokens per step and short requests finish early. Real schedulers add a token budget per step (prefill
chunks share it with decodes) and stop admitting when the KV cache is nearly full.

#### Pitfalls
- Admitting a request in static mode just because a slot is free: static mode waits for the batch to empty.
- Finishing a request at `t` instead of `t + 1`: it produces its last token *during* step `t`.
