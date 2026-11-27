# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def moe_matmul_kernel(
    x,
    w,
    y,
    order,       # token ids sorted by expert
    tile_expert, # expert of each row tile
    tile_start,  # first position in `order` of each row tile
    tile_end,    # end of that expert's group in `order`
    K,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    pid_n = tl.program_id(1)
    # Use tl.dot(a, b, input_precision="ieee"). The default TF32 is not accurate enough.


def solve(
    X: torch.Tensor,
    expert: torch.Tensor,
    W: torch.Tensor,
    Y: torch.Tensor,
    T: int,
    E: int,
    K: int,
    N: int,
) -> None:
    block_m, block_n, block_k = 64, 64, 32
    # Group the tokens by expert, then build one entry per row tile of each group.
    pass
