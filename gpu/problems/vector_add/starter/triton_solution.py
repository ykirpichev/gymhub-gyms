# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a_d = tl.load(a + offsets, mask=mask)
    b_d = tl.load(b + offsets, mask=mask)
    tl.store(c + offsets, a_d + b_d, mask=mask)


def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int) -> None:
    block = 1024
    grid = (triton.cdiv(N, block),)
    vector_add_kernel[grid](A, B, C, N, block)
