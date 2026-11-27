# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(x, gamma, beta, y, cols, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = cols


def solve(
    X: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    Y: torch.Tensor,
    rows: int,
    cols: int,
    eps: float,
) -> None:
    block = triton.next_power_of_2(cols)
    layer_norm_kernel[(rows,)](X, gamma, beta, Y, cols, eps, block)
