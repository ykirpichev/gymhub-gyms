# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(x, weight, y, cols, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < cols


def solve(
    X: torch.Tensor,
    weight: torch.Tensor,
    Y: torch.Tensor,
    rows: int,
    cols: int,
    eps: float,
) -> None:
    block = triton.next_power_of_2(cols)
    rms_norm_kernel[(rows,)](X, weight, Y, cols, eps, block)
