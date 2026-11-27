# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def int8_matvec_kernel(w, scale, x, y, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)


def solve(W: torch.Tensor, scale: torch.Tensor, x: torch.Tensor, y: torch.Tensor, N: int, K: int) -> None:
    block_n = 16
    block_k = 256
    grid = (triton.cdiv(N, block_n),)
    int8_matvec_kernel[grid](W, scale, x, y, N, K, block_n, block_k)
