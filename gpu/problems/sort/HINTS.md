# Hints for sort

Each `## Hint` section is revealed one at a time by `gpu-gym hint sort`.

## Hint 1

Start with a bitonic sort in global memory. Copy `A` into a scratch buffer of size `n = next power of two >= N` and fill the tail with `INFINITY`. Then on the host: `for (int size = 2; size <= n; size *= 2) for (int stride = size / 2; stride > 0; stride /= 2) step<<<n / 512, 256>>>(buf, size, stride);` where `step` gives each thread one pair. Copy the first `N` values to `B`.

## Hint 2

Inside `step`, thread `t` handles `i = 2 * stride * (t / stride) + t % stride` and `j = i + stride`. The pair should be ascending when `(i & size) == 0` and descending otherwise. Swap `buf[i]` and `buf[j]` when they are in the wrong order for their direction.

## Hint 3

Every stage is a kernel launch and a full pass over memory. The stages with `stride <= 1024` touch only elements inside an aligned chunk of 2048, so do them all inside one block: load the chunk into shared memory, run those stages with `__syncthreads()` between them, and write it back. That replaces the last 11 launches of every `size` with one.

## Hint 4

For radix sort, map each float to an order-preserving key with `unsigned int u = __float_as_uint(x); u ^= (u >> 31) ? 0xFFFFFFFFu : 0x80000000u;`. Each 8-bit pass is: a per-block histogram of the digit, a scan over all blocks' histograms to get each block's offset per digit, and a stable scatter in which each block writes its keys in order. Ping-pong between two buffers and undo the bit trick on the last pass.
