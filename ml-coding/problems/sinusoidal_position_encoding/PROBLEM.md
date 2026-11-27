# Sinusoidal positional encoding

Write `solve(seq_len, d_model)` that returns the Transformer's fixed positional encoding.

- `seq_len`: number of positions `L ≥ 0`.
- `d_model`: number of channels `d ≥ 1`.

Raise `ValueError` if `seq_len < 0` or `d_model < 1`.

Return a float64 NumPy array `pe` of shape `(seq_len, d_model)`. For position `pos` and pair index `i`:

```
pe[pos, 2i]     = sin(pos / 10000^(2i / d_model))
pe[pos, 2i + 1] = cos(pos / 10000^(2i / d_model))
```

Both channels of a pair share the same frequency. When `d_model` is odd, the last channel `d_model - 1` is even, so it is a sine channel with no cosine partner. `seq_len = 0` gives shape `(0, d_model)`.

```
solve(3, 4)
-> array([[ 0.    ,  1.    ,  0.    ,  1.    ],
          [ 0.8415,  0.5403,  0.01  ,  1.    ],
          [ 0.9093, -0.4161,  0.02  ,  0.9998]])
```

Channels 2 and 3 use frequency `1 / 10000^(2/4) = 0.01`.

**Interview focus:** alternating sine/cosine channels that share a frequency, the `2i / d` exponent, odd widths, and building the matrix with broadcasting instead of loops.

**Follow-ups** to answer out loud once your solution passes:

- Why can a fixed offset be expressed as a linear function of the encoding?
- How do RoPE and ALiBi differ from adding an encoding?
- How do models extend to contexts longer than they were trained on?
