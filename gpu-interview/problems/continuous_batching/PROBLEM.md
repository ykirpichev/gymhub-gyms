# Continuous versus static batching

Write `solve(requests, max_batch, mode)` that simulates a decode-only scheduler and returns when each request finishes.

- `requests`: a list of `[arrival, output_len]`: the step at which the request arrives (an int ≥ 0) and how many tokens it generates (an int ≥ 1).
- `max_batch`: the most requests that can run in one step.
- `mode`: `"continuous"` or `"static"`.

Time runs in steps `t = 0, 1, 2, …`. At the start of step `t`, the scheduler admits waiting requests that have arrived (`arrival ≤ t`), in order of arrival and then of their index in `requests`:

- `"continuous"`: admit while fewer than `max_batch` requests are running.
- `"static"`: admit only when **no** request is running, then admit up to `max_batch`.

During step `t` every running request generates one token. A request that has generated all `output_len` tokens finishes at time `t + 1` and leaves the batch. Return the list of finish times, in the order of `requests`. Raise `ValueError` for an unknown mode or `max_batch < 1`.

```
solve([[0, 3], [0, 1], [1, 1]], 2, "continuous")  ->  [3, 1, 2]
solve([[0, 3], [0, 1], [1, 1]], 2, "static")      ->  [3, 1, 4]
```

In static mode, request 2 arrives at step 1 but waits until the whole first batch is done.

#### What interviewers look for

- A clean event loop: admit, step, retire, with a well-defined order and no off-by-one in the finish time.
- Explaining the result: in static batching, slots freed by short requests sit idle until the longest request finishes.
- Knowing what this model leaves out: prefill cost, KV-cache memory as the real admission limit, and preemption.

#### Follow-ups

- Admission is limited by KV-cache blocks, not by a count. How would you combine this with the paged allocator from the previous exercise?
- A request with a 4,000-token prompt arrives. How does its prefill affect the decode latency of everyone else, and what is chunked prefill?
- How would you schedule to meet a TTFT target for interactive users while running batch jobs on the same GPUs?
