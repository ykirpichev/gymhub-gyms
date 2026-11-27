#### The idea
A naive kernel makes either the reads or the writes strided by a whole row. Stage a 32 × 32 tile in shared
memory: read it along rows of `A`, write it along rows of `C`, and do the turn in fast on-chip memory. Pad the tile
to 33 columns so reading a column of the tile does not hit one bank 32 times.

#### A solution
```cuda
#define TILE 32

__global__ void transpose_kernel(const float* A, float* C, int M, int N) {
    __shared__ float tile[TILE][TILE + 1];               // +1: no bank conflicts
    int x = blockIdx.x * TILE + threadIdx.x;              // column of A
    int y = blockIdx.y * TILE + threadIdx.y;              // row of A
    for (int j = 0; j < TILE; j += blockDim.y)
        if (y + j < M && x < N) tile[threadIdx.y + j][threadIdx.x] = A[(size_t)(y + j) * N + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;                  // column of C (a row of A)
    y = blockIdx.x * TILE + threadIdx.y;                  // row of C (a column of A)
    for (int j = 0; j < TILE; j += blockDim.y)
        if (y + j < N && x < M) C[(size_t)(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
}

extern "C" void solve(const float* A, float* C, int M, int N) {
    dim3 threads(32, 8);                                  // each thread moves 4 elements
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    transpose_kernel<<<blocks, threads>>>(A, C, M, N);
    cudaDeviceSynchronize();
}
```

#### What it costs
8 bytes per element, no math: the target is a plain copy's bandwidth. The naive kernel typically reaches a third
to a half of it; the tiled one gets within 10 to 20 percent.

#### Pitfalls
- Dropping the `__syncthreads()`: threads read tile entries other threads have not written yet.
- Swapping the block indices for the write but not the thread indices (or the reverse). Check on a non-square case:
  `M != N` catches most index mistakes.
- `[32][32]`: correct, but every column read of the tile is a 32-way bank conflict.
