# Occupancy by hand

An H100 SM has 65,536 registers and at most 64 resident warps (2048 threads) and 32 blocks. A kernel uses 256 threads
per block, 96 registers per thread and no shared memory. Registers are allocated per warp in units of 256. What is the
occupancy, in percent?
