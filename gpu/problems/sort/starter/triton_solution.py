# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def sort_step_kernel(buf, size, stride, BLOCK_SIZE: tl.constexpr):
    t = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)


def solve(A: torch.Tensor, B: torch.Tensor, N: int) -> None:
    n = triton.next_power_of_2(N)
    buf = torch.full((n,), float("inf"), device=A.device, dtype=A.dtype)
    buf[:N] = A
    block = 1024
    # for size in 2, 4, ..., n: for stride in size // 2, ..., 1: launch one step.
    B.copy_(buf[:N])
