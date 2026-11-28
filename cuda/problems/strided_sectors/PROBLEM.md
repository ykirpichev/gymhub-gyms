# Sectors of a strided load

A warp of 32 threads executes `v = x[threadIdx.x * 2]` on a `float` array aligned to 128 bytes. How many 32-byte sectors
does the load touch?
