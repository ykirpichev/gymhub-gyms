# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def top_k_kernel(x, y, cols, k, BLOCK_SIZE: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    # Walk the row in chunks: for start in range(0, cols, BLOCK_SIZE): ...


def solve(X: torch.Tensor, Y: torch.Tensor, rows: int, cols: int, k: int) -> None:
    block = 1024
    block_k = triton.next_power_of_2(k)
    top_k_kernel[(rows,)](X, Y, cols, k, block, block_k)
