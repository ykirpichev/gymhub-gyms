// GPU_GYM_STARTER
#include <cuda_runtime.h>

// Y[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))), x = X[i].
__global__ void gelu_kernel(const float* X, float* Y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
    }
}

extern "C" void solve(const float* X, float* Y, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(X, Y, N);
    cudaDeviceSynchronize();
}
