// GPU_GYM_STARTER
#include <cuda_runtime.h>

// X and Y are T x H x D, cos and sin are T x (D / 2). For i < D / 2, with a = X[t, h, i] and
// b = X[t, h, i + D / 2]:  Y[t, h, i] = a * cos - b * sin,  Y[t, h, i + D / 2] = b * cos + a * sin.
__global__ void rope_kernel(const float* X, const float* cos, const float* sin, float* Y, int T,
                            int H, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < T * H * D) {
    }
}

extern "C" void solve(const float* X, const float* cos, const float* sin, float* Y, int T, int H,
                      int D) {
    int threads = 256;
    int blocks = (T * H * D + threads - 1) / threads;
    rope_kernel<<<blocks, threads>>>(X, cos, sin, Y, T, H, D);
    cudaDeviceSynchronize();
}
