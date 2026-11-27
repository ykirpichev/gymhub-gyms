// GPU_GYM_STARTER
#include <cuda_runtime.h>

// A is row-major M x K. y[r] = sum over k of A[r * K + k] * x[k].
__global__ void matvec_kernel(const float* A, const float* x, float* y, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
    }
}

extern "C" void solve(const float* A, const float* x, float* y, int M, int K) {
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    matvec_kernel<<<blocks, threads>>>(A, x, y, M, K);
    cudaDeviceSynchronize();
}
