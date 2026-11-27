# Int8 weight matvec

**Level:** medium   **Teaches:** mixed-type loads, dequantization in registers, memory-bound GEMV with 4× fewer weight bytes

## Task

Multiply a weight matrix stored as **int8**, with one FP32 scale per row, by an FP16 vector, and write the FP32 result. This is the inner loop of a weight-quantized LLM decoding one token at a time: the weights are read once per token, so storing them in 8 bits instead of 16 cuts the time almost in half.

## Arguments

| Name    | Type            | Shape   | Role   |
| ------- | --------------- | ------- | ------ |
| `W`     | `const int8_t*` | `N × K` | input  |
| `scale` | `const float*`  | `N`     | input  |
| `x`     | `const __half*` | `K`     | input  |
| `y`     | `float*`        | `N`     | output |
| `N`     | `int`           |         | size   |
| `K`     | `int`           |         | size   |

`W` is row-major: row `n` holds the weights of output `n`, as in `nn.Linear`. In PyTorch and Triton, `W` is `torch.int8`, `x` is `torch.float16`, and `scale` and `y` are `torch.float32`.

## Formula

```
y[n] = scale[n] * sum over k of float(W[n, k]) * float(x[k])     accumulated in FP32
```

## Example

```
W = [[1, -2, 3], [127, 0, -127]]   scale = [0.5, 0.01]   x = [1, 0.5, -1]
y = [0.5 * (1 - 1 - 3), 0.01 * (127 + 127)] = [-1.5, 2.54]
```

## How to think about it

Start from `matvec`: one warp (or block) per row, lanes striding along `K` so loads coalesce, then a warp-shuffle reduction. Three things change:

1. **Mixed types.** Convert each weight with `(float)W[...]` and each input with `__half2float(x[...])`, multiply and add in FP32. Apply `scale[n]` once, after the reduction, not to every product.
2. **Bytes.** The matrix now takes one byte per weight, so reading it at full bandwidth needs wide loads: a lane that loads one `int8_t` at a time issues 4 times the instructions of the float version. Load 4 weights as one `int` (or 16 as an `int4`/`uint4` vector) and unpack them in registers. `x` is small and reused by every row: read it from L1/L2 or stage it in shared memory.
3. **The roofline.** The benchmark moves about `N × K` bytes, so the bound is a quarter of an FP32 matvec of the same shape. Compare your GB/s with `gpu-gym bench`'s copy peak.

In Triton: one program per block of rows, load `W` blocks with `tl.load(...).to(tl.float32)`, `x` blocks with `.to(tl.float32)`, accumulate `tl.sum(w * x[None, :], axis=1)` over `K`, then multiply by the scales.

## In an interview

Weight-only quantization questions test whether you know decoding is memory-bound and what the dequantization costs.

Follow-ups to answer out loud once your kernel passes:

- Why does int8 storage speed up batch-1 decoding, but not a large-batch prefill?
- How would you change the kernel for int4 weights packed two per byte, with a scale per group of 128?
- Where would you apply a zero point for asymmetric quantization, and what does it cost per element?
- At what batch size would you switch from this GEMV to a GEMM kernel?
