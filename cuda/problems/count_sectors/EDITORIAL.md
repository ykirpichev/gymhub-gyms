#### Idea
The DRAM and L2 move whole sectors. Map every byte range to its sectors and count the distinct ones; efficiency
compares useful bytes with moved bytes.

#### What the numbers say
A coalesced float load moves 4 sectors per warp. Stride 2 moves 8, stride 32 moves 32: the kernel then needs up to 8×
the DRAM bandwidth for the same work. A misaligned but contiguous access costs only one extra sector, which is why
alignment matters far less than contiguity on modern GPUs.

#### Pitfalls
- Counting 128-byte lines instead of sectors: since Volta the L2 and DRAM transfer granularity is 32 bytes.
- Forgetting that a read can straddle a sector boundary.
