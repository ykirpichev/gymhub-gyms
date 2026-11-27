# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(logits, targets, loss, V, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    # Walk the row in chunks: for start in range(0, V, BLOCK_SIZE): ...


def solve(logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor, rows: int, V: int) -> None:
    block = 4096
    cross_entropy_kernel[(rows,)](logits, targets, loss, V, block)
