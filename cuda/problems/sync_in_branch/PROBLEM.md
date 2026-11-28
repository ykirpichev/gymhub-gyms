# A __syncthreads in a branch

A kernel does this with 256 threads per block:

```cuda
if (threadIdx.x < 128) {
    s[threadIdx.x] = x[threadIdx.x];
    __syncthreads();
    y[threadIdx.x] = s[127 - threadIdx.x];
}
```

What is wrong, and what is the fix?
