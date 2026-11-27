# Simulate a ring all-reduce

`N` ranks each hold a buffer of length `L`, divisible by `N`. Split each buffer into `N` chunks of `L / N` elements; chunk `c` is elements `c·L/N` to `(c+1)·L/N − 1`. Rank `r` sends only to rank `(r + 1) % N`.

**Reduce-scatter**: for step `s = 0, 1, …, N − 2`, every rank `r` sends its chunk `(r − s) % N` to the next rank, which **adds** it into its own copy of that chunk. After these `N − 1` steps, rank `r` holds the full sum of chunk `(r + 1) % N`.

**All-gather**: for step `s = 0, 1, …, N − 2`, every rank `r` sends its chunk `(r + 1 − s) % N` to the next rank, which **overwrites** its copy of that chunk. After these steps every rank holds the full sum everywhere.

In each step all ranks send at the same time, using the values from *before* the step.

Write `solve(buffers)` that takes a list of `N` buffers (lists of numbers) and returns a tuple `(after_reduce_scatter, result, sent)`:

- `after_reduce_scatter`: every rank's buffer after the reduce-scatter phase (a list of `N` lists).
- `result`: every rank's buffer at the end.
- `sent`: the number of elements each rank sent in total.

```
solve([[1, 2], [10, 20]])
-> ([[1, 22], [11, 20]], [[11, 22], [11, 22]], 2)
```

With `N = 2`, step 0 of reduce-scatter has rank 0 send chunk 0 to rank 1, and rank 1 send chunk 1 to rank 0.

Raise `ValueError` if the buffers differ in length or the length is not divisible by `N`. With one rank, nothing is sent.

#### What interviewers look for

- A schedule where every link is busy every step and no rank sends more than its share.
- Taking a snapshot per step, so ranks don't read values updated in the same step.
- The communication volume, $2 \frac{N-1}{N}$ times the buffer, and why it does not grow with `N`.

#### Follow-ups

- The ring takes `2(N − 1)` steps, each paying a latency. When is a tree all-reduce better? (Small messages, many ranks.)
- How does a hierarchical all-reduce use fast NVLink inside a node and slower networking between nodes?
- How do reduce-scatter and all-gather map onto ZeRO stage 3?
