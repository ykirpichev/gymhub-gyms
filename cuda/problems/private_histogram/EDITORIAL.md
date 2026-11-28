#### Idea
Each block accumulates privately, then merges. Shared-memory atomics are cheap (on-chip, and contention stays inside one
block); global atomics are reduced to at most `bins` per block.

#### When it pays
With skewed data (many equal values) a naive global histogram serializes thousands of atomics on one address. The
private version absorbs that in shared memory. For many bins (say 65,536) the private histogram no longer fits in
shared memory; then sort the keys first, or use per-warp histograms of a sub-range.

#### Pitfalls
- Forgetting to zero the shared histogram before use (shared memory is not initialized), and the `__syncthreads()`
  between zeroing, counting and merging.
