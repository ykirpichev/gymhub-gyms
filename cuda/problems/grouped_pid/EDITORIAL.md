#### Idea
Tiles are visited in vertical strips `group_m` tiles tall: down the strip, then one column right. Programs launched
together cover a compact `group_m × k` patch of C, which reads `group_m` row panels of A and `k` column panels of B,
instead of 1 row panel and many column panels for row-by-row order.

#### Why it matters
For a large GEMM the inputs do not fit in L2. Row-major order streams all of B through L2 for every row of tiles;
grouped order reuses each B panel across `group_m` rows while it is still cached. The tutorial reports around 10% more
throughput from this alone. CUTLASS calls the same idea a *swizzled tile scheduler*; Hopper kernels combine it with
persistent blocks that loop over tiles.

#### Pitfalls
- The last group can be shorter than `group_m` when `num_pid_m` is not a multiple: that is what `group_size_m` handles.
