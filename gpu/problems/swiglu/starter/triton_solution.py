# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(x, y, cols, BLOCK_SIZE: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < cols


def solve(X: torch.Tensor, Y: torch.Tensor, rows: int, cols: int) -> None:
    block = 1024
    grid = (rows, triton.cdiv(cols, block))
    swiglu_kernel[grid](X, Y, cols, block)
