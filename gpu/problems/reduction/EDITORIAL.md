#### The idea
Three levels, each with the cheapest tool for it:
1. **Thread:** a grid-stride loop, so each thread adds many elements into a register while the loads stream in
   coalesced.
2. **Block:** warp shuffles reduce 32 values without shared memory; one value per warp goes through shared memory.
3. **Grid:** one `atomicAdd` per block, or per-block partials and a second tiny launch if the result must be the same
   on every run.

#### A solution
```cuda
__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// Every thread gets the block's total. blockDim.x is a multiple of 32.
__device__ float block_sum(float v) {
    __shared__ float part[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_sum(v);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    v = threadIdx.x < blockDim.x / 32 ? part[threadIdx.x] : 0.0f;
    if (warp == 0) v = warp_sum(v);
    if (threadIdx.x == 0) part[0] = v;
    __syncthreads();
    float total = part[0];
    __syncthreads();                            // part[] is reused by the next call
    return total;
}

__global__ void reduce_kernel(const float* A, float* out, int N) {
    float v = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        v += A[i];
    v = block_sum(v);
    if (threadIdx.x == 0) atomicAdd(out, v);
}

extern "C" void solve(const float* A, float* out, int N) {
    int threads = 256, sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int blocks = min((N + threads - 1) / threads, 8 * sms);   // enough to fill the GPU, no more
    cudaMemset(out, 0, sizeof(float));                          // out starts as NaN
    reduce_kernel<<<blocks, threads>>>(A, out, N);
    cudaDeviceSynchronize();
}
```
PyTorch: `out.copy_(A.sum().reshape(1))`. Triton: per-program `tl.sum` and one `tl.atomic_add` per program.

#### What it costs
`4N` bytes read and nothing else worth counting: the bound is `4N / bandwidth`. A good kernel reaches 80 to 90
percent of it. Each step of the ladder fixes one bottleneck:

| Version | Bottleneck it removes |
|---|---|
| `atomicAdd(out, A[i])` per element | (baseline) millions of serialized atomics on one address |
| shared-memory tree, one atomic per block | atomics; now limited by `__syncthreads()` and idle threads |
| grid-stride loop, fewer blocks | per-block overhead; most threads now do loads, not tree steps |
| warp shuffles for the last levels | shared-memory traffic and syncs |
| `float4` loads | instruction count per byte |

#### Pitfalls
- Not zeroing `out` first: it is filled with NaN before your call.
- `__syncthreads()` inside `if (i < N)`: some threads never reach it. Keep every thread in the reduction and feed
  out-of-range threads a 0.
- `if (tid % (2 * s) == 0)` (interleaved addressing) works but diverges inside every warp; keep active threads
  contiguous (`tid < s`), as in the diagram above.
- Atomics make the float result depend on the order blocks finish, so two runs can differ in the last bits. If the
  interviewer asks for determinism, write per-block partials and reduce them in a second launch.
