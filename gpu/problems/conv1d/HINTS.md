# Hints for 1D convolution

Each `## Hint` section is revealed one at a time by `gpu-gym hint conv1d`.

## Hint 1

Inside the starter's `if (i < out_n)` block, loop `k` from 0 to `K`, accumulate `A[i + k] * W[k]` into a local `float sum = 0.0f`, then write `C[i] = sum` once. Do not accumulate with `C[i] +=`: the output starts out filled with NaN.

Triton: `k` is a runtime value, and `for j in range(0, k):` works. Inside it, load `a + index + j` with the mask and `w + j` as a scalar, and accumulate into `tl.zeros([BLOCK_SIZE], dtype=tl.float32)`. PyTorch: `torch.nn.functional.conv1d` expects `(batch, channels, length)` shapes, so reshape with `view(1, 1, -1)`.

## Hint 2

All threads read the same `W[k]` at the same time. Put `W` in `__constant__` memory, which broadcasts one address to a whole warp. Declare `__constant__ float Wc[64];` and copy into it in `solve` with `cudaMemcpyToSymbol(Wc, W, K * sizeof(float), 0, cudaMemcpyDeviceToDevice)`. Keep the plain-pointer path for `K` larger than the array.

## Hint 3

A block of `T` threads needs `T + K - 1` consecutive inputs. Load them into shared memory once, with coalesced reads, then `__syncthreads()` and compute every window from shared memory. The extra `K - 1` "halo" values need a second load by some threads. Guard the loads near the end of `A`.

## Hint 4

Have each thread compute several adjacent outputs, for example 4 or 8. It loads `outputs + K - 1` inputs into registers once and reuses each loaded value for up to `K` multiply-adds. Combine this with `float4` loads of the input slice. Reduce the grid in `solve` to match.
