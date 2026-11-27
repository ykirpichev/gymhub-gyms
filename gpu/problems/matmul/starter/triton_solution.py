# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a,
    b,
    c,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)


def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int) -> None:
    block_m, block_n, block_k = 64, 64, 32
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    matmul_kernel[grid](A, B, C, M, N, K, block_m, block_n, block_k)
