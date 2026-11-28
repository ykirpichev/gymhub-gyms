# Steps of a tree reduction

A block of 1024 threads sums 1024 values in shared memory with a tree, halving the active threads each step. How many
steps (each followed by `__syncthreads()`) does it take?
