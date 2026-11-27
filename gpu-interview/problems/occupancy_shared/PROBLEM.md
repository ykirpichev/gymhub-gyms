# Occupancy limited by shared memory

An SM has 100 KB (102,400 bytes) of shared memory for blocks and can hold at most 2,048 threads. A kernel uses 128 threads per block, few registers, and 40 KB (40,960 bytes) of shared memory per block.

What is its occupancy, as a percentage of 2,048 threads?
