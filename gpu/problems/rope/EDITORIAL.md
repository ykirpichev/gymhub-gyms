#### The idea
Each pair `(i, i + D/2)` of each head of each token is rotated by an angle that depends only on the token and `i`.
One thread per pair: `T × H × D/2` threads, each reading two values and one `cos`/`sin` pair, writing two values.

#### A solution
```cuda
__global__ void rope_kernel(const float* X, const float* cosT, const float* sinT, float* Y,
                            int T, int H, int D) {
    int half = D / 2;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)T * H * half) return;
    int i = idx % half;
    size_t th = idx / half;                    // token * H + head
    int t = th / H;
    const float* x = X + th * D;
    float a = x[i], b = x[i + half];
    float c = cosT[(size_t)t * half + i], s = sinT[(size_t)t * half + i];
    Y[th * D + i] = a * c - b * s;
    Y[th * D + i + half] = b * c + a * s;
}
```
Launch `ceil(T * H * D / 2 / 256)` blocks of 256. PyTorch: split `X` into halves `x1, x2` along the last axis,
broadcast `cos`/`sin` over heads with `[:, None, :]`, and write `cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)`.

#### What it costs
8 bytes per element of `X` plus the tables, which every head of a token shares (cache hits). Memory-bound, and in
practice fused into the QKV projection or the attention kernel.

#### Pitfalls
- Pairing neighbours `(2i, 2i + 1)`: that is the other RoPE layout (GPT-J style). This one is "rotate half".
- The sign: `a·c − b·s` for the first half, `b·c + a·s` for the second.
- Indexing the tables by head: all heads of a token share the same angles.
