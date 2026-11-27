// GPU_GYM_STARTER
#include <cuda_runtime.h>

// Change this signature together with problem.py if your exercise is not vector add.
__global__ void kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    kernel<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
