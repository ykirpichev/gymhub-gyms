#### The idea
One thread per element again; `alpha` arrives by value, so every thread already has it in a register.

#### A solution
```cuda
__global__ void saxpy_kernel(const float* A, const float* B, float* C, float alpha, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = fmaf(alpha, A[i], B[i]);    // one fused multiply-add
}
```
PyTorch: `torch.add(B, A, alpha=alpha, out=C)`.

#### What it costs
12 bytes and 2 FLOPs per element: memory-bound, the same time as `vector_add`. That is the point of the exercise:
the extra multiply is free.

#### Pitfalls
- Treating `alpha` as a pointer (`*alpha`) or copying it to the device: it is a plain `float` argument.
- `fmaf` rounds once instead of twice, so results can differ from PyTorch in the last bit. The tolerance allows it.
