# A barrier inside a branch

A block of 128 threads runs:

```cuda
if (threadIdx.x < 64) {
    tile[threadIdx.x] = x[i];
    __syncthreads();
}
y[i] = tile[127 - threadIdx.x];
```

What is the problem?
