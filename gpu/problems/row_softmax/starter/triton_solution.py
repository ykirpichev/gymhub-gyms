# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x, y, cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = cols


def solve(X: torch.Tensor, Y: torch.Tensor, rows: int, cols: int) -> None:
    block = triton.next_power_of_2(cols)
    softmax_kernel[(rows,)](X, Y, cols, block)
