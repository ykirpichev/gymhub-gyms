#### Idea
A launch creates `grid × block` threads with global indices `0 … grid·block − 1`. With one element per thread, element
`j` belongs to thread `j` if that thread exists. With a grid-stride loop each thread starts at its index and jumps by
the grid's total size, so element `j` belongs to thread `j mod (grid·block)`.

#### Why grid-stride loops matter
The grid size no longer depends on `n`: launch a few blocks per SM (say `4 × number of SMs` blocks) and let the loop
cover the rest. That amortizes per-thread setup, keeps every SM busy, and the same kernel works for any size.

#### Pitfalls
- Forgetting the bounds check: the last block reads past the end of the array.
- Computing the stride as `blockDim.x` instead of `gridDim.x * blockDim.x`: threads of different blocks then repeat
  each other's work.
