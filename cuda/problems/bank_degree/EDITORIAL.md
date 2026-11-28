#### Idea
Group the words by bank; within a bank, count distinct words. The slowest bank decides how many passes the access
takes.

#### Rules of thumb
- Stride `s` gives a `gcd(s, 32)`-way conflict for a full warp: odd strides are free.
- 8-byte and 16-byte accesses (`double`, `float4`) are split into phases of 16 or 8 lanes by the hardware, so
  `float4` reads of consecutive elements are conflict-free.
- Swizzling (XOR-ing the column with the row) is the padding-free alternative that CUTLASS and TMA use.
