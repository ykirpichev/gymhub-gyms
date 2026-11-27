# Hints for 2D convolution

Each `## Hint` section is revealed one at a time by `gpu-gym hint conv2d`.

## Hint 1

Get a correct version first. In the starter each thread owns one output `(row, col)`. Inside the `if`, loop over `u` and `v` and add `A[(row + u) * W + (col + v)] * F[u * KW + v]`. Write the sum to `C[row * OW + col]`, where `OW = W - KW + 1`, not `W`.

## Hint 2

Put the filter in constant memory. Declare `__constant__ float F_c[15 * 15];` at file scope. In `solve`, copy the filter in before the launch: `cudaMemcpyToSymbol(F_c, F, KH * KW * sizeof(float), 0, cudaMemcpyDeviceToDevice);` (`F` is a device pointer, so the kind must be device to device). Read `F_c[u * KW + v]` in the kernel.

## Hint 3

Load a tile into shared memory. With 16 × 16 threads, the block needs `(16 + KH - 1) × (16 + KW - 1)` inputs, which is more than 256, so each thread loads several with a loop: `for (int k = tid; k < TH * TW; k += 256)`. Guard the loads at the right and bottom edges of the image. Call `__syncthreads()`, then compute from the tile. Pad the tile row (`TW + 1`) if you see bank conflicts.

## Hint 4

Have each thread compute several outputs, for example 4 or 8 in the same column or row. Neighbouring outputs share most of their inputs, so a thread can keep a sliding window of input values in registers and reuse each load several times. Fewer, larger blocks also mean a smaller share of the tile is halo.
