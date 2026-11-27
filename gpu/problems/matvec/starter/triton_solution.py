# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_kernel(a, x, y, M, K, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M


def solve(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, M: int, K: int) -> None:
    block_m = 16
    block_k = 128
    grid = (triton.cdiv(M, block_m),)
    matvec_kernel[grid](A, x, y, M, K, block_m, block_k)
