// GPU_GYM_STARTER
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

// A is M x K and B is K x N, FP16, row-major. C is M x N, FP32, row-major. M, N, K are multiples of 16.
__global__ void gemm_fp16_kernel(const __half* A, const __half* B, float* C, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
    }
}

extern "C" void solve(const __half* A, const __half* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    gemm_fp16_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
