#### Idea
A layout is a function from a coordinate to an offset. CuTe accepts a 1D index for any shape and splits it with the
first mode fastest (colexicographic order, like column-major), recursively for nested modes; each leaf contributes
`coordinate × stride`.

#### Why CuTe works this way
One algebra describes everything: row- and column-major matrices, tiles of tiles, how 128 threads own the fragments of a
`wgmma` accumulator, and swizzled shared memory. Partitioning a tensor over threads is composing layouts, so a kernel's
data movement is written once and checked at compile time.

#### Pitfalls
- Splitting the index lexicographically (last mode fastest): CuTe's default is the reverse.
- Treating a nested mode's size as its first number: the size of `[2, 2]` is 4.
