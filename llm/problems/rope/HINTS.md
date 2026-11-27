# Hints for RoPE

## Hint 1

`inv_freq = 10000 ** (-torch.arange(half) * 2 / Dh)` and `pos = torch.arange(start, start + T)`. The outer product `pos[:, None] * inv_freq[None, :]` is the `[T, half]` angle table. Make both on `q.device`.

## Hint 2

Split with `x1, x2 = q[..., :half], q[..., half:]`. A `[T, half]` table broadcasts against `[B, H, T, half]` with no reshaping. Glue the two halves back with `torch.cat(..., dim=-1)`.

## Hint 3

For speed, the `cos`/`sin` table is tiny; the cost is reading and writing `q`. `torch.cat` and the four products each make a full-size temporary. Write into a preallocated output (`out[..., :half] = ...`), or use `torch.compile`, or a Triton kernel that handles both halves of a row in one program.
