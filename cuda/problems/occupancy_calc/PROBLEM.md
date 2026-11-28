# Occupancy calculator

Write a simplified occupancy calculator. Inputs: `threads` per block, `regs` per thread, `smem` bytes of shared memory
per block, and the SM's limits `max_warps`, `max_blocks`, `regs_per_sm`, `smem_per_sm` (bytes).

Rules:
- warps per block = `ceil(threads / 32)`.
- Registers are allocated per warp, rounded up to a multiple of 256: `regs_per_warp = ceil(regs * 32 / 256) * 256`.
  Blocks allowed by registers = `(regs_per_sm // regs_per_warp) // warps_per_block`.
- Blocks allowed by warps = `max_warps // warps_per_block`; by block slots = `max_blocks`.
- Blocks allowed by shared memory = `smem_per_sm // smem` (no limit when `smem` is 0).

Return `[blocks_per_sm, occupancy]` where occupancy = resident warps / `max_warps`, rounded to 4 decimals.

Example: `solve(256, 64, 49152, 64, 32, 65536, 233472)` (H100 limits, 48 KB of shared memory): warps 8, registers allow
4 blocks, warps 8, block slots 32, shared memory 4 → `[4, 0.5]`.
