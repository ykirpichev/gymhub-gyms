// GPU_GYM_STARTER
#include <cuda_runtime.h>

// A is H x W, F is KH x KW, C is (H - KH + 1) x (W - KW + 1). All row-major.
// C[i, j] = sum over u, v of A[i + u, j + v] * F[u, v]
__global__ void conv2d_kernel(const float* A, const float* F, float* C, int H, int W, int KH,
                              int KW) {
    int OH = H - KH + 1;
    int OW = W - KW + 1;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < OH && col < OW) {
    }
}

extern "C" void solve(const float* A, const float* F, float* C, int H, int W, int KH, int KW) {
    int OH = H - KH + 1;
    int OW = W - KW + 1;
    dim3 threads(16, 16);
    dim3 blocks((OW + threads.x - 1) / threads.x, (OH + threads.y - 1) / threads.y);
    conv2d_kernel<<<blocks, threads>>>(A, F, C, H, W, KH, KW);
    cudaDeviceSynchronize();
}
