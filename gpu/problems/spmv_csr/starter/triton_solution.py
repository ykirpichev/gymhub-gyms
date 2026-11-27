# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def spmv_kernel(row_ptr, col_idx, values, x, y, BLOCK_SIZE: tl.constexpr):
    r = tl.program_id(0)
    start = tl.load(row_ptr + r).to(tl.int32)
    end = tl.load(row_ptr + r + 1).to(tl.int32)


def solve(
    row_ptr: torch.Tensor,
    col_idx: torch.Tensor,
    values: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    rows: int,
    cols: int,
    nnz: int,
) -> None:
    block = 128
    spmv_kernel[(rows,)](row_ptr, col_idx, values, x, y, block)
