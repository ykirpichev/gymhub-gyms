#include <cuda_runtime.h>

// C has length N - K + 1. C[i] = sum_k A[i + k] * W[k]
__global__ void conv1d_kernel(const float* A, const float* W, float* C, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int out_n = N - K + 1;
    if (i < out_n) {
        float accum = 0.0f; 
        for (int k = 0; k < K; k++) {
            accum += A[i + k] * W[k];
        }
        C[i] = accum;
    }
}

extern "C" void solve(const float* A, const float* W, float* C, int N, int K) {
    int out_n = N - K + 1;
    int threads = 256;
    int blocks = (out_n + threads - 1) / threads;
    conv1d_kernel<<<blocks, threads>>>(A, W, C, N, K);
    cudaDeviceSynchronize();
}
