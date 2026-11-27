#### The idea
The naive kernel does one global `atomicAdd` per element. When many elements share a bin, those atomics collide and
serialize. Give each block a private histogram in shared memory (`bins <= 1024`, at most 4 KB), count into it with
fast shared atomics, then merge it into the global one with one atomic per non-empty bin.

#### A solution
```cuda
__global__ void hist_kernel(const float* A, float* counts, int N, int bins) {
    extern __shared__ unsigned int local[];
    for (int b = threadIdx.x; b < bins; b += blockDim.x) local[b] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        atomicAdd(&local[(int)A[i]], 1u);
    __syncthreads();
    for (int b = threadIdx.x; b < bins; b += blockDim.x)
        if (local[b]) atomicAdd(&counts[b], (float)local[b]);
}

extern "C" void solve(const float* A, float* counts, int N, int bins) {
    int threads = 256, sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    cudaMemset(counts, 0, bins * sizeof(float));          // all-zero bits is 0.0f
    hist_kernel<<<4 * sms, threads, bins * sizeof(unsigned int)>>>(A, counts, N, bins);
    cudaDeviceSynchronize();
}
```
PyTorch: `counts.copy_(torch.bincount(A.long(), minlength=bins).float())`.

#### What it costs
`4N` bytes read; the rest is atomics. With a private histogram, global atomics drop from `N` to about
`blocks × bins`. Skewed data (everything in one bin) is the stress test: shared atomics still collide within a block,
and a further trick is one histogram per warp.

#### Pitfalls
- Forgetting to zero `counts` (it starts as NaN) or the shared histogram (shared memory is not initialized).
- Counting in `float` in shared memory works, but integer counters are exact by construction and faster.
- Launching one block per 256 elements: then each block merges a whole histogram for very little counting.
