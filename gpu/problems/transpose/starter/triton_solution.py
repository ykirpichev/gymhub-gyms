# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(a, c, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)


def solve(A: torch.Tensor, C: torch.Tensor, M: int, N: int) -> None:
    block_m = 32
    block_n = 32
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    transpose_kernel[grid](A, C, M, N, block_m, block_n)
