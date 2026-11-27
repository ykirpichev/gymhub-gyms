#### The idea
A gather: row `i` of the output is a copy of row `ids[i]` of the table. Give each token a block and let its threads
copy the `D` values of the row, so reads and writes are both contiguous.

#### A solution
```cuda
__global__ void embedding_kernel(const float* ids, const float* table, float* out, int D) {
    int i = blockIdx.x;
    const float* src = table + (size_t)(int)ids[i] * D;
    float* dst = out + (size_t)i * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) dst[d] = src[d];
}

extern "C" void solve(const float* ids, const float* table, float* out, int N, int V, int D) {
    embedding_kernel<<<N, 128>>>(ids, table, out, D);
    cudaDeviceSynchronize();
}
```
PyTorch: `out.copy_(table[ids.long()])` or `F.embedding`.

#### What it costs
`8ND` bytes (read a row, write a row): a pure copy, memory-bound. Copy with `float4` when `D % 4 == 0` to cut the
instruction count by four. Repeated ids hit L2, so the real read traffic can be lower.

#### Pitfalls
- Indexing `table` with the float directly: convert with `(int)ids[i]` first.
- `int` overflow in `ids[i] * D` for large tables; do the multiply in `size_t`.
- One thread per token copying a whole row in a loop: each warp then reads 32 different rows at once, uncoalesced.
