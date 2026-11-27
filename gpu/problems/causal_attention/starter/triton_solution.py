# GPU_GYM_STARTER
import torch
import triton
import triton.language as tl


@triton.jit
def causal_attention_kernel(q, k, v, o, L, d, scale,
                            BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    qt = tl.program_id(0)  # query tile
    h = tl.program_id(1)   # head
    # Use tl.dot(a, b, input_precision="ieee"). The default TF32 is not accurate enough.


def solve(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor, H: int, L: int, d: int
) -> None:
    block_n, block_m = 64, 64
    block_d = max(16, triton.next_power_of_2(d))
    scale = 1.0 / d**0.5
    grid = (triton.cdiv(L, block_n), H)
    causal_attention_kernel[grid](Q, K, V, O, L, d, scale, block_n, block_m, block_d)
