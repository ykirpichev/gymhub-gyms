# Hints for gemm_fp16

Each `## Hint` section is revealed one at a time by `gpu-gym hint gemm_fp16`.

## Hint 1

Get a correct version first, without Tensor Cores: one thread per output, `float acc = 0.0f;`, then `acc += __half2float(A[m * K + k]) * __half2float(B[k * N + n]);`. In PyTorch, `C.copy_(torch.matmul(A.float(), B.float()))` is correct but casts first; `torch.matmul(A, B).float()` uses Tensor Cores.

## Hint 2

WMMA: launch one warp per 16 × 16 output tile. In the kernel, `wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;`, the same with `matrix_b`, and `wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;` set to zero with `fill_fragment`. For `k0 = 0, 16, ...`: `load_matrix_sync(a, A + tile_m * 16 * K + k0, K)`, `load_matrix_sync(b, B + k0 * N + tile_n * 16, N)`, `mma_sync(c, a, b, c)`. Finish with `store_matrix_sync(C + tile_m * 16 * N + tile_n * 16, c, N, wmma::mem_row_major)`.

## Hint 3

Several warps per block, each with a 2 × 2 grid of output fragments (32 × 32 of `C`), sharing tiles of `A` and `B` staged in shared memory: one global load of a tile feeds several `mma_sync` calls. Pad the shared tiles (for example by 8 halves per row) to avoid bank conflicts in `load_matrix_sync`.

## Hint 4

Triton: grid over `(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))`, offsets `rm`, `rn`, `rk`; loop over K loading `a = tl.load(A + rm[:, None] * K + (k + rk)[None, :])` and `b` likewise, `acc += tl.dot(a, b)`, and store `acc` into `C`. Start with `BLOCK_M = BLOCK_N = 128`, `BLOCK_K = 32`, `num_warps = 4`, `num_stages = 3`, then tune.
