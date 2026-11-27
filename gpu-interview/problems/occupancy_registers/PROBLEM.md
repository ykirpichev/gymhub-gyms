# Occupancy limited by registers

An SM has 65,536 32-bit registers and can hold at most 2,048 threads and 32 blocks. A kernel is launched with 256 threads per block and uses 64 registers per thread. It uses no shared memory.

What is its **occupancy**: the resident threads per SM as a percentage of the 2,048 maximum? Ignore allocation granularity.
