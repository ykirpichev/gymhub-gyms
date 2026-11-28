# Read a profile

Nsight Compute reports, for a LayerNorm kernel on an H100: DRAM throughput 31% of peak, compute (SM) throughput 18% of
peak, achieved occupancy 12% (theoretical 12%), 255 registers per thread, block size 256, top warp stall reason
"long scoreboard" (waiting on global memory). Each block handles one row of 4096 BF16 values.

Explain what limits this kernel, and propose two concrete changes, saying why each would help.
