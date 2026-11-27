## Hint 1
`H_out = (H + 2*padding - kH) // stride + 1`, and the same for `W`. If either size is below 1, the kernel does not fit: raise `ValueError` rather than returning an empty array.

## Hint 2
`np.pad` the two spatial axes only. The window for output `(i, j)` starts at row `i * stride` and column `j * stride` of the padded image. Dot that window with `weight[c]` as stored — do not flip it. A 180° rotation is a different operation and fails on an asymmetric kernel.

## Hint 3
Sum over input channels, kernel rows and kernel columns, then add `bias[c]` once. One output channel is independent of the others, so a loop over `c` with a vectorised window sum is enough.
