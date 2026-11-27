// GPU_GYM_STARTER
#include <cuda_runtime.h>
#include <math.h>

// Q, K, V, O are H x L x d, row-major. For each head h and row i:
// O[h, i] = softmax over j <= i of (Q[h, i] . K[h, j] / sqrt(d)), applied to the rows V[h, j].
__global__ void causal_attention_kernel(const float* Q, const float* K, const float* V, float* O,
                                        int H, int L, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // row = h * L + i
    if (row < H * L) {
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* O, int H, int L, int d) {
    int threads = 128;
    int blocks = (H * L + threads - 1) / threads;
    causal_attention_kernel<<<blocks, threads>>>(Q, K, V, O, H, L, d);
    cudaDeviceSynchronize();
}
