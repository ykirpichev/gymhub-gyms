# A lower bound for RMSNorm

An RMSNorm kernel reads an FP32 matrix `X` of shape 8192 × 4096 and a weight vector of 4,096 floats, and writes `Y` of the same shape as `X`. At 3 TB/s (3 × 10¹² bytes per second), what is the lowest time it can take, in **microseconds**?
