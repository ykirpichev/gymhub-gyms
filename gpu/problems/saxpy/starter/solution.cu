// GPU_GYM_STARTER
#include <cuda_runtime.h>

__global__ void saxpy_kernel(const float* A, const float* B, float* C, float alpha, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
    }
}

extern "C" void solve(const float* A, const float* B, float* C, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    saxpy_kernel<<<blocks, threads>>>(A, B, C, alpha, N);
    cudaDeviceSynchronize();
}
