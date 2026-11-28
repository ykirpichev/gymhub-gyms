# Predict a kernel's time

Write a roofline time estimate. Given a kernel's `flops` and DRAM `bytes`, the GPU's `peak_tflops` (TFLOP/s) and
`bw_tbs` (TB/s), return `[time_us, bound]`: the lower-bound time in microseconds, `max(flops / peak, bytes / bandwidth)`,
rounded to 2 decimals, and `"memory"` or `"compute"`, whichever term is larger (`"memory"` on a tie).

Example: a 4096 × 4096 BF16 GEMM on an H100 (`flops = 137438953472`, `bytes = 100663296`, 989 TFLOP/s, 3.35 TB/s):
compute 138.97 µs, memory 30.05 µs → `[138.97, "compute"]`.
