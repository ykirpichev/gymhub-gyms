// GPU_GYM_STARTER
#include <cuda_runtime.h>

__global__ void reduce_kernel(const float* A, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
    }
}

extern "C" void solve(const float* A, float* out, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    reduce_kernel<<<blocks, threads>>>(A, out, N);
    cudaDeviceSynchronize();
}
