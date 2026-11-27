// GPU_GYM_STARTER
#include <cuda_runtime.h>

// B[i] = A[N - 1 - i]. A is not changed.
__global__ void reverse_kernel(const float* A, float* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
    }
}

extern "C" void solve(const float* A, float* B, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    reverse_kernel<<<blocks, threads>>>(A, B, N);
    cudaDeviceSynchronize();
}
