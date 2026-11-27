# Hints for embedding lookup

Each `## Hint` section is revealed one at a time by `gpu-gym hint embedding`.

## Hint 1

Get it right first: with one thread per output element, `i = idx / D` and `d = idx % D`. Read `int row = (int)ids[i];` and write `out[idx] = table[row * D + d];`. Consecutive threads now share `i` and walk `d`, so the warp reads one contiguous piece of a row.

## Hint 2

Launch one block per token: `grid = N`, `i = blockIdx.x`. Read `ids[i]` once per block, then loop `for (int d = threadIdx.x; d < D; d += blockDim.x)`. This saves the division and modulo on every element.

## Hint 3

When `D % 4 == 0`, copy `float4`s: cast both row pointers with `reinterpret_cast<const float4*>` and loop to `D / 4`. Each thread then moves 16 bytes per load. Keep a scalar path for other `D`.

## Hint 4

In Triton, give each program one token: load `row = tl.load(ids + i).to(tl.int32)`, then `tl.load(table + row * D + offs, mask=offs < D)` and store it to `out + i * D + offs`, with `BLOCK_SIZE = triton.next_power_of_2(D)`.
