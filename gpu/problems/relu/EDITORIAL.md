#### The idea
The same one-thread-per-element pattern as `vector_add`, with `fmaxf` as the operation.

#### A solution
```cuda
__global__ void relu_kernel(const float* A, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = fmaxf(A[i], 0.0f);
}
```
The starter's `solve` already launches it with a rounded-up grid. PyTorch: `torch.clamp_min(A, 0, out=C)` or
`C.copy_(torch.relu(A))`. Triton: `tl.maximum(x, 0.0)`.

#### What it costs
8 bytes per element (read 4, write 4) and almost no math, so the kernel runs at memory speed. Branching
(`A[i] > 0 ? A[i] : 0`) gives the same speed here: the compiler turns it into a select, and even divergent lanes would
be hidden behind memory latency.

#### Pitfalls
- `max(A[i], 0)` with an integer `0` picks the integer overload on some compilers; write `0.0f`.
- In PyTorch, `C = torch.relu(A)` rebinds the local name and leaves the output tensor untouched. Write into `C`.
