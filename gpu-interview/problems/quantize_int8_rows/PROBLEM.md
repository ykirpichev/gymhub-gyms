# Symmetric int8 quantization per row

Write `solve(W)` that quantizes a matrix (a list of rows of floats) to int8 with one scale per row:

- `scale[r] = max(|W[r, :]|) / 127`. If a row is all zeros, its scale is `1.0`.
- `q[r, c] = round(W[r, c] / scale[r])`, rounding half to even (NumPy's `np.round` and Python's `round` both do), then clipped to the range −127 to 127.

Return a tuple `(q, scales)`: `q` as a list of lists of ints (or an integer array), `scales` as a list of floats.

```
solve([[0.5, -1.0, 0.25], [0.0, 0.0, 0.0]])
-> ([[64, -127, 32], [0, 0, 0]], [0.007874, 1.0])
```

`0.5 / (1/127) = 63.5`, which rounds to the even 64.

#### What interviewers look for

- Per-row scales, and why they beat one scale for the whole matrix: one large weight no longer crushes every other row's resolution.
- Using −127 to 127, not −128: keeping the range symmetric makes `−q` exact.
- Thinking about the zero row and about error: the worst-case error per value is `scale / 2`.

#### Follow-ups

- When would you use asymmetric quantization (a zero point)? (Activations after ReLU, which are all positive.)
- Why are LLM *activations* harder to quantize than weights? (Outlier channels; see LLM.int8() and SmoothQuant.)
- How would you quantize in groups of 128 columns instead of whole rows, and what does it cost in storage?
