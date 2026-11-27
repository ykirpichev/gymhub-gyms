# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(x, cos, sin, y, H, D, BLOCK_H: tl.constexpr, BLOCK_HALF: tl.constexpr):
    t = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)[:, None]
    i = tl.arange(0, BLOCK_HALF)[None, :]
    half = D // 2


def solve(
    X: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    Y: torch.Tensor,
    T: int,
    H: int,
    D: int,
) -> None:
    block_h = triton.next_power_of_2(H)
    block_half = triton.next_power_of_2(D // 2)
    rope_kernel[(T,)](X, cos, sin, Y, H, D, block_h, block_half)
