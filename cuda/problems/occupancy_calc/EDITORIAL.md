#### Idea
Each resource gives an upper bound on resident blocks; the SM can hold the smallest of them. The real calculator (in
Nsight Compute, or `cudaOccupancyMaxActiveBlocksPerMultiprocessor`) adds details: shared memory is allocated in
128-byte units with 1 KB reserved per block, the L1/shared split is chosen from a few carve-outs, and registers are
allocated per warp in units of 256.

#### Pitfalls
- Dividing registers per block instead of per warp: allocation granularity is per warp.
- Forgetting that blocks are all or nothing: 21 warps of registers with 8-warp blocks hold 2 blocks, not 2.6.

#### Use it in practice
`cudaOccupancyMaxPotentialBlockSize` suggests a block size, and `__launch_bounds__(threads, minBlocks)` tells the
compiler to keep registers low enough for `minBlocks` blocks per SM.
