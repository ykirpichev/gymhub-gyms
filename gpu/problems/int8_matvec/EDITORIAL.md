#### The idea
`matvec` with 1-byte weights. Batch-1 decoding is memory-bound on the weights, so reading 1 byte per weight instead of
2 (FP16) or 4 (FP32) is the whole gain. Keep the structure: one warp per output row, lanes read consecutive bytes,
accumulate `float(w) · float(x)` in FP32, apply the row's scale once at the end.

#### A solution
```cuda
#include <cuda_fp16.h>
__device__ float warp_sum(float v) {          // lane 0 ends up with the warp's total
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

__global__ void int8_matvec_kernel(const int8_t* W, const float* scale, const __half* x, float* y, int N, int K) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= N) return;
    const int8_t* w = W + (size_t)row * K;
    float s = 0.0f;
    for (int k = lane; k < K; k += 32) s += (float)w[k] * __half2float(x[k]);
    s = warp_sum(s);
    if (lane == 0) y[row] = scale[row] * s;
}
// launch: 8 warps per block, (N + 7) / 8 blocks
```
PyTorch: `y.copy_(scale * (W.float() @ x.float()))`, which is correct but materializes an FP32 copy of `W`.

#### What it costs
`N · K` bytes of weights dominate. Byte-wide loads mean each warp instruction fetches only 32 bytes, so the next step
is loading 4 or 16 weights per lane at once (`char4` or `int4` vector loads) and converting them in registers. With
the input also quantized, the `__dp4a` instruction does four int8 multiply-adds into an int32 in one go.

#### Pitfalls
- Applying the scale per element inside the loop: correct but wasted multiplies; it factors out of the sum.
- Accumulating in FP16: overflow and rounding for long rows.
- Treating `W` as unsigned: it is signed int8 (−128 to 127).
