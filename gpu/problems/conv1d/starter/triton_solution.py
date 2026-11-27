import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(a, w, c, n, k, BLOCK_SIZE: tl.constexpr):
    index = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_n = n - k + 1
    mask = index < out_n
    accum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for j in range(k):
        accum += tl.load(a + index + j, mask=mask, other=0.0) * tl.load(w + j)
    tl.store(c + index, accum, mask=mask)


def solve(A: torch.Tensor, W: torch.Tensor, C: torch.Tensor, N: int, K: int) -> None:
    block = 128
    out_n = N - K + 1
    grid = (triton.cdiv(out_n, block),)
    conv1d_kernel[grid](A, W, C, N, K, tl.constexpr(block))
