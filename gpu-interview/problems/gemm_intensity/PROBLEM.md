# Intensity of a large matmul

`C = A @ B` with `A`, `B` and `C` all 4096 × 4096 in BF16 (2 bytes per element). With the minimum traffic (read `A` and `B` once, write `C` once), what is its arithmetic intensity in FLOPs per byte?
