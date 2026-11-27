# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(a, f, c, W, OH, OW, KH: tl.constexpr, KW: tl.constexpr,
                  BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)


def solve(
    A: torch.Tensor, F: torch.Tensor, C: torch.Tensor, H: int, W: int, KH: int, KW: int
) -> None:
    OH, OW = H - KH + 1, W - KW + 1
    block_h, block_w = 16, 64
    grid = (triton.cdiv(OH, block_h), triton.cdiv(OW, block_w))
    conv2d_kernel[grid](A, F, C, W, OH, OW, KH, KW, block_h, block_w)
