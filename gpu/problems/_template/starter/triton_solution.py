# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements


def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int) -> None:
    block = 1024
    grid = (triton.cdiv(N, block),)
    kernel[grid](A, B, C, N, block)
