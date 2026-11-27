#### The idea
The 2D version of `conv1d`. A block computes a tile of outputs (say 16 × 16), so it needs an input tile of
`(16 + KH - 1) × (16 + KW - 1)` values. Load that into shared memory once, load the filter (at most 15 × 15) once,
then each thread computes one output from shared memory.

#### A solution
```cuda
#define TO 16                                   // output tile
#define MAXK 15

__global__ void conv2d_kernel(const float* A, const float* F, float* C, int H, int W, int KH, int KW) {
    __shared__ float in[TO + MAXK - 1][TO + MAXK - 1];
    __shared__ float f[MAXK * MAXK];
    int OH = H - KH + 1, OW = W - KW + 1;
    int r0 = blockIdx.y * TO, c0 = blockIdx.x * TO;
    int th = TO + KH - 1, tw = TO + KW - 1;
    int tid = threadIdx.y * TO + threadIdx.x;
    for (int t = tid; t < th * tw; t += TO * TO) {
        int r = r0 + t / tw, c = c0 + t % tw;
        in[t / tw][t % tw] = (r < H && c < W) ? A[(size_t)r * W + c] : 0.0f;
    }
    for (int t = tid; t < KH * KW; t += TO * TO) f[t] = F[t];
    __syncthreads();
    int i = r0 + threadIdx.y, j = c0 + threadIdx.x;
    if (i < OH && j < OW) {
        float acc = 0.0f;
        for (int u = 0; u < KH; ++u)
            for (int v = 0; v < KW; ++v)
                acc += in[threadIdx.y + u][threadIdx.x + v] * f[u * KW + v];
        C[(size_t)i * OW + j] = acc;
    }
}

extern "C" void solve(const float* A, const float* F, float* C, int H, int W, int KH, int KW) {
    dim3 threads(TO, TO);
    dim3 blocks((W - KW + 1 + TO - 1) / TO, (H - KH + 1 + TO - 1) / TO);
    conv2d_kernel<<<blocks, threads>>>(A, F, C, H, W, KH, KW);
    cudaDeviceSynchronize();
}
```

#### What it costs
`2 · KH · KW` FLOPs per output. With a 3 × 3 filter it is memory-bound; with 15 × 15 it is 450 FLOPs per output and
compute-bound, and the shared tile is what keeps it so: without it, every input is fetched up to 225 times.

#### Pitfalls
- Output index arithmetic: `C` has `OW = W - KW + 1` columns, not `W`.
- Loading the tile with one element per thread only covers 16 × 16; the halo needs the strided loop above.
- Real CNN layers have channels and batches, and libraries turn them into a GEMM (im2col) or use Winograd/FFT.
  Mention it if asked how cuDNN does it.
