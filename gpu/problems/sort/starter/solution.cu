// GPU_GYM_STARTER
#include <cuda_runtime.h>
#include <math.h>

// B = A sorted in ascending order. N is any size; pad to a power of two with INFINITY if you need to.
// You can allocate scratch memory with cudaMalloc and launch as many kernels as you like.
__global__ void sort_step_kernel(float* buf, int n, int size, int stride) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n / 2) {
    }
}

extern "C" void solve(const float* A, float* B, int N) {
    cudaMemcpy(B, A, N * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}
