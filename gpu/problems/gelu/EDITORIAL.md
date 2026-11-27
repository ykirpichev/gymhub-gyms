#### The idea
Elementwise again, now with real math per element. Write the formula once with the constants as `float` literals.

#### A solution
```cuda
__global__ void gelu_kernel(const float* X, float* Y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = X[i];
        float u = 0.7978845608f * (x + 0.044715f * x * x * x);
        Y[i] = 0.5f * x * (1.0f + tanhf(u));
    }
}
```
PyTorch: `Y.copy_(torch.nn.functional.gelu(X, approximate="tanh"))`.

#### What it costs
About 15 FLOPs including `tanhf` per 8 bytes: still memory-bound on any modern GPU, whose ridge point is above
10 FLOPs per byte for FP32. This is why frameworks fuse activations into the matmul before them: on its own, GELU
costs a full read and write of the tensor.

#### Pitfalls
- `0.044715` without the `f` suffix is a `double`, and the whole expression is computed in FP64, which is many times
  slower on consumer GPUs.
- Using the exact erf form (`normcdf`) when the task asks for the tanh approximation fails the tolerance near 0.
