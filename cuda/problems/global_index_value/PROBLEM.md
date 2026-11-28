# Which thread owns the element?

A 1D kernel is launched with `<<<64, 128>>>` and computes `i = blockIdx.x * blockDim.x + threadIdx.x`. Which
element `i` does thread 37 of block 5 handle?
