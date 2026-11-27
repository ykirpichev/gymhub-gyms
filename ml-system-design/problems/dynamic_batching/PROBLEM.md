# A dynamic batching server

Classic model servers (for classifiers, embedders, rankers) batch requests with two knobs: `max_batch` and `max_wait`. Write `solve(arrivals, max_batch, max_wait, base_ms, per_item_ms)` that simulates one worker and returns each request's latency.

- `arrivals`: request arrival times in ms, sorted ascending (floats or ints, possibly equal).
- `max_batch`: an int `≥ 1`. `max_wait`: ms, `≥ 0`.
- A batch of `n` requests takes `base_ms + per_item_ms * n` ms to run. The worker runs one batch at a time.

Requests are served in arrival order. Let `i` be the oldest request not yet served and `free` the time the worker finishes its current batch (0 at the start). The next batch starts at

```
start = max(free, arrivals[i], min(time the max_batch-th waiting request arrives, arrivals[i] + max_wait))
```

where "the `max_batch`-th waiting request" is request `i + max_batch − 1` (infinity if it does not exist). In words: launch as soon as the batch is full or the oldest request has waited `max_wait`, but never before the worker is free. The batch holds every request with `arrival ≤ start`, from `i` onward, at most `max_batch` of them. Each one finishes at `start + base_ms + per_item_ms * n`.

Return a list of latencies (finish time minus arrival), one per request in order.

```
solve([0, 1, 2, 10], 2, 5, 3, 1)  ->  [6, 5, 9, 9]
```

Requests 0 and 1 fill a batch at time 1 and finish at 6. Request 2 waits until its deadline at 7 (request 3 arrives too late), finishing at 11. Request 3 arrives at 10, the worker is free at 11, but the batch is not full, so it waits until 15 and finishes at 19.

#### What interviewers look for

- An event loop with a clear rule for when a batch launches, and correct handling of a busy worker (requests that arrived while it was busy can launch at once).
- Explaining the trade-off: a larger `max_wait` builds fuller batches (higher throughput per batch) and adds up to `max_wait` of latency at low traffic.
- Knowing this is not how LLM generation is batched (outputs differ in length, so LLM servers batch per decode step).

#### Follow-ups

- How would you pick `max_batch` and `max_wait` for a p99 latency target of 50 ms?
- At what arrival rate does this server become unstable (the queue grows without bound)?
- How would you extend it to several workers, or to GPU batches whose cost depends on the padded sequence length?
