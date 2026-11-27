#### The idea
Elementwise with a twist in the layout: the gate and the up value for `(r, c)` sit `cols` apart in the same row.
One thread per output element reads both and writes one.

#### A solution
```cuda
__global__ void swiglu_kernel(const float* X, float* Y, int rows, int cols) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)rows * cols) return;
    size_t r = idx / cols, c = idx % cols;
    const float* x = X + r * 2 * cols;
    float g = x[c], u = x[cols + c];
    Y[idx] = g / (1.0f + expf(-g)) * u;
}
```
PyTorch: `g, u = X.chunk(2, dim=-1); Y.copy_(F.silu(g) * u)`.

#### What it costs
12 bytes per output (two reads, one write) and a handful of FLOPs: memory-bound. Both reads coalesce, since
consecutive threads read consecutive `c` in each half. In a real model this activation is fused into the epilogue of
the matmul that produced `X`, so the `rows × 2·cols` intermediate never goes to memory.

#### Pitfalls
- Reading the up value from the next row (`X[(r + 1) * cols + c]`) instead of the second half of the same row.
- `silu(g) = g * sigmoid(g)`; using `sigmoid(g)` alone gives the GLU variant, not SwiGLU.
