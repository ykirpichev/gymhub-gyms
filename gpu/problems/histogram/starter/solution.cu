// GPU_GYM_STARTER
#include <cuda_runtime.h>

// A holds N whole numbers in [0, bins) stored as floats.
// counts[b] = how many times b appears in A.
__global__ void histogram_kernel(const float* A, float* counts, int N, int bins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
    }
}

extern "C" void solve(const float* A, float* counts, int N, int bins) {
    // counts starts filled with NaN. Zero it before adding to it.
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    histogram_kernel<<<blocks, threads>>>(A, counts, N, bins);
    cudaDeviceSynchronize();
}
