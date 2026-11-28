# Row-major GEMM with cuBLAS

Your matrices are row-major: `A` is `M × K`, `B` is `K × N`, and you want row-major `C = A·B` (`M × N`). cuBLAS assumes
column-major. Which call computes it without copying or transposing any data?
