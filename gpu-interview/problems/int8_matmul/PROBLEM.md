# An int8 matmul with rescaling

A W8A8 linear layer stores activations `xq` (an `M × K` int8 matrix with one scale per row, `xs`) and weights `wq` (an `N × K` int8 matrix, one row per output feature as in `nn.Linear`, with one scale per row, `ws`). Write `solve(xq, xs, wq, ws)` that returns a tuple `(acc, y)`:

- `acc`: the `M × N` integer matrix `xq @ wq.T`, computed in integers (as int32 accumulation on a GPU).
- `y`: the `M × N` float matrix `acc[i, j] * xs[i] * ws[j]`.

```
solve([[1, -2]], [0.5], [[3, 4], [-1, 0]], [0.1, 0.2])
-> ([[-5, -1]], [[-0.25, -0.1]])
```

#### What interviewers look for

- Doing the multiply in integers and scaling once per output, not dequantizing every input element first.
- That per-row activation scales and per-row weight scales factor out of the sum over `K`; a scale per *column of `K`* would not.
- Knowing the overflow bound: each product is at most 127 × 127 = 16,129, so int32 holds sums over `K` up to about 133,000.

#### Follow-ups

- Why can't you use per-`K`-group scales with a plain integer GEMM, and what do group-quantized kernels do instead?
- A weight-only int4 kernel (W4A16) never multiplies integers. What does it do in the inner loop, and why is it still faster for decoding?
- Where would you fuse the rescale (and a bias or activation) in a GPU kernel?
