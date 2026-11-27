# Intensity of a vector add

A kernel computes `c[i] = a[i] + b[i]` for FP32 arrays. What is its arithmetic intensity in FLOPs per byte of DRAM traffic? Count only the minimum traffic: each input read once, the output written once.
