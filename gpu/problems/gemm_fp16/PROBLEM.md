# FP16 matmul on Tensor Cores

**Level:** hard   **Teaches:** Tensor Cores (WMMA or `tl.dot`), mixed precision, FP32 accumulation, compute-bound tiling

## Task

Multiply two FP16 matrices and write the product in FP32: `C = A @ B`. This is the core of every transformer layer in training and inference. On NVIDIA GPUs since Volta, **Tensor Cores** do a small matrix multiply-accumulate (for example 16 × 16 × 16) in one instruction for a whole warp, an order of magnitude faster than the FP32 CUDA cores of the `matmul` exercise.

## Arguments

| Name | Type            | Shape   | Role   |
| ---- | --------------- | ------- | ------ |
| `A`  | `const __half*` | `M × K` | input  |
| `B`  | `const __half*` | `K × N` | input  |
| `C`  | `float*`        | `M × N` | output |
| `M`  | `int`           |         | size   |
| `N`  | `int`           |         | size   |
| `K`  | `int`           |         | size   |

All matrices are row-major. In every test, `M`, `N` and `K` are multiples of 16. In PyTorch and Triton, `A` and `B` arrive as `torch.float16` tensors and `C` as `torch.float32`.

## Formula

```
C[m, n] = sum over k of float(A[m, k]) * float(B[k, n])     accumulated in FP32
```

## Example

```
A = [[1, 2], [3, 4]]  (FP16)     B = [[0.5, 0], [0, 0.25]]  (FP16)
C = [[0.5, 0.5], [1.5, 1.0]]     (FP32)
```

(The tests use multiples of 16; the example is small to read.)

## How to think about it

The product of two FP16 numbers is exact in FP32, so the only rounding allowed is in the sum. **Accumulate in FP32.** A kernel that sums in FP16 fails the larger tests: with `K = 4096`, the running sum's FP16 rounding error is far above the tolerance. Tensor Core instructions take FP16 inputs and keep an FP32 accumulator, which is exactly this contract.

In CUDA, start with the **WMMA API** (`#include <mma.h>`, namespace `nvcuda::wmma`): each warp owns one 16 × 16 tile of `C` as an accumulator *fragment*, loops over `K` in steps of 16, loads a 16 × 16 tile of `A` and of `B` into matrix fragments with `load_matrix_sync`, calls `mma_sync`, and finally stores with `store_matrix_sync`. That alone beats the FP32 `matmul` many times over. Then apply what `matmul` taught: stage tiles of `A` and `B` in shared memory so several warps reuse them, give each warp several output tiles, and double-buffer.

In Triton, `tl.dot` on FP16 blocks uses Tensor Cores and accumulates in FP32 when you pass an FP32 accumulator: `acc += tl.dot(a, b)` with `acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)`.

The roofline: `4096³` takes `2 × 4096³ = 137` GFLOP. Compare your TFLOP/s with your GPU's FP16 Tensor Core peak, not its FP32 peak; `gpu-gym bench` also compares with `torch.matmul`, which calls cuBLAS.

## In an interview

Tensor Core questions check that you know the precision contract (FP16 or BF16 in, FP32 accumulate) and the data movement around the instruction.

Follow-ups to answer out loud once your kernel passes:

- Why are the products exact in FP32, and where does the error come from?
- What would change for BF16 inputs, and for FP8 with per-tensor scales?
- How would you handle `M`, `N` or `K` that are not multiples of 16?
- Your kernel reaches 40% of the Tensor Core peak. What limits it? (Shared-memory bandwidth, too few warps in flight, no double buffering, the epilogue.)
