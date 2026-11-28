#### Idea
Count tiles in each dimension, rounding up, and compare the work done with the work needed.

#### What it shows
A large GEMM fills every instruction. A decode step (`M` = batch size, often 1 to 16) uses a few percent of each MMA,
which is fine because decode is memory-bound anyway: the matrix must be read from HBM once whatever the tile shape.
Kernels for small `M` therefore swap operands (compute `Cᵀ = Bᵀ Aᵀ` so the big dimension maps to the tile's `m`),
or use split-K to spread the work over more SMs.

#### Pitfalls
- Integer division without rounding up: the remainder tiles still cost a full instruction.
