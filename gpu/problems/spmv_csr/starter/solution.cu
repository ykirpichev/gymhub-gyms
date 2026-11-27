// GPU_GYM_STARTER
#include <cuda_runtime.h>

// CSR matrix: the nonzeros of row r are p in [row_ptr[r], row_ptr[r + 1]), at column col_idx[p] with
// value values[p]. row_ptr and col_idx hold whole numbers stored as floats. y[r] = sum values[p] * x[col].
__global__ void spmv_kernel(const float* row_ptr, const float* col_idx, const float* values,
                            const float* x, float* y, int rows) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
    }
}

extern "C" void solve(const float* row_ptr, const float* col_idx, const float* values,
                      const float* x, float* y, int rows, int cols, int nnz) {
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    spmv_kernel<<<blocks, threads>>>(row_ptr, col_idx, values, x, y, rows);
    cudaDeviceSynchronize();
}
