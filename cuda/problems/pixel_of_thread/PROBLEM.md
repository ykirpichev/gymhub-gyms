# 2D thread to pixel

An image of `height × width` pixels is stored row-major (`index = row * width + col`). A kernel is launched with
`dim3 block(bx, by)` and a rounded-up grid, and computes `col = blockIdx.x * bx + threadIdx.x`,
`row = blockIdx.y * by + threadIdx.y`.

Write `solve(width, height, bx, by, block_x, block_y, thread_x, thread_y)` that returns the linear index of the pixel
that thread `(thread_x, thread_y)` of block `(block_x, block_y)` processes, or `-1` if it falls outside the image.

Example: a 100 × 50 image (width 100, height 50) with 16 × 16 blocks; thread (3, 2) of block (1, 2) has
`col = 19`, `row = 34`, index `34 * 100 + 19 = 3419`.
