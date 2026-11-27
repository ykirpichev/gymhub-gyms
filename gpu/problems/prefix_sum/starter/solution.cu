// GPU_GYM_STARTER
#include <cuda_runtime.h>

// Inclusive scan: B[i] = A[0] + ... + A[i].
// Start with one thread and a plain loop. Then split the work across blocks.
__global__ void scan_kernel(const float* A, float* B, int N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
    }
}

extern "C" void solve(const float* A, float* B, int N) {
    scan_kernel<<<1, 1>>>(A, B, N);
    cudaDeviceSynchronize();
}
