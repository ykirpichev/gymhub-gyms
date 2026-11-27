#### The idea
A scan has a dependency chain, but it can be done in three data-parallel passes:
1. Each block scans its own 1,024 elements (warp shuffles inside a warp, then a scan of the 32 warp totals) and
   writes its block total.
2. Scan the block totals: the same kernel, recursively.
3. Add each block's offset (the scanned total of all blocks before it) to its elements.

#### A solution
```cuda
__device__ float block_scan(float v, float* warp_tot) {     // inclusive; blockDim.x == 1024
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int o = 1; o < 32; o <<= 1) {
        float n = __shfl_up_sync(0xffffffff, v, o);
        if (lane >= o) v += n;
    }
    if (lane == 31) warp_tot[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = warp_tot[lane];
        for (int o = 1; o < 32; o <<= 1) {
            float n = __shfl_up_sync(0xffffffff, t, o);
            if (lane >= o) t += n;
        }
        warp_tot[lane] = t;
    }
    __syncthreads();
    return warp > 0 ? v + warp_tot[warp - 1] : v;
}

__global__ void scan_blocks(const float* in, float* out, float* sums, int n) {
    __shared__ float warp_tot[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = block_scan(i < n ? in[i] : 0.0f, warp_tot);   // every thread takes part
    if (i < n) out[i] = v;
    if (threadIdx.x == blockDim.x - 1) sums[blockIdx.x] = v;
}

__global__ void add_offsets(float* out, const float* sums, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && i < n) out[i] += sums[blockIdx.x - 1];
}

static void scan(const float* in, float* out, int n) {
    int blocks = (n + 1023) / 1024;
    float* sums;
    cudaMalloc(&sums, blocks * sizeof(float));
    scan_blocks<<<blocks, 1024>>>(in, out, sums, n);
    if (blocks > 1) {
        scan(sums, sums, blocks);           // in place: each thread reads its element before writing it
        add_offsets<<<blocks, 1024>>>(out, sums, n);
    }
    cudaFree(sums);
}

extern "C" void solve(const float* A, float* B, int N) {
    scan(A, B, N);
    cudaDeviceSynchronize();
}
```
PyTorch: `torch.cumsum(A, 0, out=B)`.

#### What it costs
This version reads `A` once, writes `B`, and re-reads and re-writes `B` in pass 3: about 16 bytes per element
instead of the minimum 8. Production scans (CUB's *decoupled look-back*) do it in one pass: each block publishes its
total as soon as it knows it, and the next block waits only for its predecessor's prefix.

#### Pitfalls
- An exclusive scan by mistake (shifted by one): the task wants `B[i]` to include `A[i]`.
- `__shfl_up_sync` returns the lane's own value when `lane < o`; the `if (lane >= o)` guard is what makes it right.
- Hillis–Steele in shared memory without double buffering reads values that were updated in the same step.
- `cudaMalloc` inside the timed call costs time; allocate once if you optimize further.
