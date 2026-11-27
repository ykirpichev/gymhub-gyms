#### The idea
Every output depends on one index, so give each thread one index. The grid is rounded up to whole blocks, so the
last block has threads past the end: they must do nothing.

#### A solution
```cuda
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;   // round up
    vector_add<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

PyTorch: `torch.add(A, B, out=C)`. Triton: one program per `BLOCK_SIZE` elements, `tl.load(..., mask=offs < N)`,
add, `tl.store(..., mask=...)`.

#### What it costs
12 bytes of traffic per element (two reads, one write) and one FLOP: arithmetic intensity 1/12, deep in memory-bound
territory. The best possible time is `12 N / bandwidth`; for `N = 2^24` on a 2 TB/s card that is about 0.1 ms. The
benchmark's GB/s figure tells you how close you got.

#### Pitfalls
- Forgetting `if (i < N)`: the extra threads of the last block write past the end of `C`.
- `N / threads` without rounding up drops the tail when `N` is not a multiple of the block size.
- Block sizes that are not a multiple of 32 waste lanes of the last warp. 128 to 512 is a good range.
