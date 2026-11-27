// GPU_GYM_STARTER
#include <cuda_runtime.h>
#include <math.h>

// Q is N x d, K and V are M x d, O is N x d. All row-major.
// O = softmax(Q K^T / sqrt(d)) V, softmax over the M keys of each query row.
__global__ void attention_kernel(const float* Q, const float* K, const float* V, float* O, int N,
                                 int M, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float scale = 1.0f / sqrtf((float)d);
    if (i < N) {
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* O, int N, int M,
                      int d) {
    int threads = 128;
    int blocks = (N + threads - 1) / threads;
    attention_kernel<<<blocks, threads>>>(Q, K, V, O, N, M, d);
    cudaDeviceSynchronize();
}
