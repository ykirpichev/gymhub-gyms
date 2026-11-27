# What fusion saves

Unfused, `y = gelu(x + b)` on FP32 runs as two kernels: `t = x + b`, then `y = gelu(t)`. Each kernel reads its input from HBM and writes its output to HBM; ignore the bias vector, which is tiny. Fused, one kernel reads `x` and writes `y`. Both versions are memory-bound and run at the same bandwidth. How many times faster is the fused version?
