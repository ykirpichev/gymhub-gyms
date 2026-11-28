# Unified memory surprises

A program allocates 4 GB with `cudaMallocManaged`, fills it on the CPU, then launches a kernel that reads it once. The
kernel is ten times slower than the same kernel with `cudaMalloc` and an explicit copy. Which explanations and fixes are
right?
