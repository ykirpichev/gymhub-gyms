# Hints for rotary embeddings

Each `## Hint` section is revealed one at a time by `gpu-gym hint rope`.

## Hint 1

Use one thread per pair. With `half = D / 2` and `idx` running over `T * H * half`, split it as `i = idx % half`, `th = idx / half` (the `(t, h)` row) and `t = th / H`. The pair is `X[th * D + i]` and `X[th * D + i + half]`, and the angles are `cos[t * half + i]` and `sin[t * half + i]`.

## Hint 2

The starter launches one thread per element of `X`, which is twice as many as there are pairs. Change it to `T * H * (D / 2)` threads, or keep the launch and let only the first half of each head do the work.

## Hint 3

Use a 2D grid to avoid the divisions: `blockIdx.x` is the token, `threadIdx.y` walks the heads and `threadIdx.x` walks `i`. Load the token's `cos` and `sin` row into shared memory once per block and let all heads read it from there.

## Hint 4

In Triton, give each program one token and all its heads: `h = tl.arange(0, BLOCK_H)[:, None]`, `i = tl.arange(0, BLOCK_HALF)[None, :]`. Load `c` and `s` as `(1, BLOCK_HALF)` rows and let broadcasting apply them to every head.
