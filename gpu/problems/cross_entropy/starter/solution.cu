// GPU_GYM_STARTER
#include <cuda_runtime.h>
#include <math.h>

// logits is rows x V, row-major. targets holds whole numbers in [0, V) stored as floats.
// loss[r] = log(sum_j exp(logits[r, j])) - logits[r, targets[r]]. Subtract the row max first.
__global__ void cross_entropy_kernel(const float* logits, const float* targets, float* loss,
                                     int rows, int V) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
    }
}

extern "C" void solve(const float* logits, const float* targets, float* loss, int rows, int V) {
    int threads = 128;
    int blocks = (rows + threads - 1) / threads;
    cross_entropy_kernel<<<blocks, threads>>>(logits, targets, loss, rows, V);
    cudaDeviceSynchronize();
}
