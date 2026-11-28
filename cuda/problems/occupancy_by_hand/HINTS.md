## Hint 1

Registers per warp: 96 × 32 = 3072, already a multiple of 256.

## Hint 2

Warps that fit by registers: 65,536 / 3072 = 21.3, so 21. Blocks are 8 warps each.
