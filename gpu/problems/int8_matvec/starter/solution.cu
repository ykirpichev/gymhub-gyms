// GPU_GYM_STARTER
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// W is N x K int8, row-major. scale is N floats, x is K halves, y is N floats.
__global__ void int8_matvec_kernel(const int8_t* W, const float* scale, const __half* x, float* y, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
    }
}

extern "C" void solve(const int8_t* W, const float* scale, const __half* x, float* y, int N, int K) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int8_matvec_kernel<<<blocks, threads>>>(W, scale, x, y, N, K);
    cudaDeviceSynchronize();
}
