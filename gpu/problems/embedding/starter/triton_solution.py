# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(ids, table, out, D, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D


def solve(ids: torch.Tensor, table: torch.Tensor, out: torch.Tensor, N: int, V: int, D: int) -> None:
    block = triton.next_power_of_2(D)
    embedding_kernel[(N,)](ids, table, out, D, block)
