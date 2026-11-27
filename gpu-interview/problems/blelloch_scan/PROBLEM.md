# Work-efficient scan

An exclusive scan of `[3, 1, 7, 0, 4, 1, 6, 3]` is `[0, 3, 4, 11, 11, 15, 16, 22]`: each output is the sum of the inputs before it. The work-efficient (Blelloch) scan does it in $O(n)$ additions and $2 \log_2 n$ parallel steps, in place:

**Up-sweep** (reduce). For `stride = 1, 2, 4, …, n/2`: for every `i = 2·stride − 1, 4·stride − 1, …` below `n`, `a[i] += a[i − stride]`. Afterwards the last element holds the total, and the array holds a tree of partial sums.

**Down-sweep**. Set `a[n − 1] = 0`. For `stride = n/2, …, 2, 1`: for every `i = 2·stride − 1, 4·stride − 1, …` below `n`, `t = a[i − stride]`, `a[i − stride] = a[i]`, `a[i] += t`. Afterwards `a` is the exclusive scan.

Write `solve(values)` for a list whose length is a power of two (1, 2, 4, …). Return a tuple `(tree, scan)`: the array after the up-sweep, and the exclusive scan, both as lists.

```
solve([3, 1, 7, 0, 4, 1, 6, 3])  ->  ([3, 4, 7, 11, 4, 5, 6, 25], [0, 3, 4, 11, 11, 15, 16, 22])
```

Raise `ValueError` if the length is 0 or not a power of two.

#### What interviewers look for

- The two phases, and the swap-and-add in the down-sweep, without looking them up.
- Why this matters on a GPU: the naive (Hillis–Steele) scan does $O(n \log n)$ additions; this one does $O(n)$ with the same number of steps, doubled.
- How it extends past one block: scan each block, scan the block totals, add them back.

#### Follow-ups

- In the CUDA version, where do the `__syncthreads()` go, and why does the access pattern cause bank conflicts?
- How would you handle a length that is not a power of two?
- What else is a scan used for? (Stream compaction, radix sort, the sampler in top-p.)
