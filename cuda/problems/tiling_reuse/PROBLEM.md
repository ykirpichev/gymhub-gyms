# Why tiling helps

A tiled matrix multiply of two 4096 × 4096 matrices uses 32 × 32 tiles in shared memory. How many times is each element
of A loaded from global memory (ignoring caches)? A naive kernel loads it 4096 times.
