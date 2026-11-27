# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(a, counts, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    N = N


def solve(A: torch.Tensor, counts: torch.Tensor, N: int, bins: int) -> None:
    # counts starts filled with NaN. Zero it before adding to it.
    block = 1024
    grid = (triton.cdiv(N, block),)
    histogram_kernel[grid](A, counts, N, block)
