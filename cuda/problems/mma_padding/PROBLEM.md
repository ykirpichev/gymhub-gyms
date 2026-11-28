# Tensor Core tiles and padding

A GEMM of shape `M × N × K` (C is `M × N`) runs on Tensor Core instructions that each compute an `m × n × k` tile.
Dimensions that are not multiples of the tile are padded up.

Return `[instructions, utilization]`: the number of MMA instructions
(`ceil(M/m) * ceil(N/n) * ceil(K/k)`), and the useful fraction of their work, `M*N*K / (instructions * m*n*k)`,
rounded to 4 decimals.

Example: `solve(4096, 4096, 4096, 64, 256, 16)` (Hopper `wgmma` m64n256k16): 64 × 16 × 256 = 262,144 instructions,
utilization `1.0`. A decode step with one token, `solve(1, 4096, 4096, 64, 256, 16)`, issues 4096 instructions at
utilization `0.0156`.
