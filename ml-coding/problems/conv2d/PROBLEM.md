# 2D convolution

Write `solve(x, weight, bias, stride, padding)`.

The operation is **cross-correlation**, which is what a deep-learning `conv2d` computes. Do not rotate the filter.

- `x`: floats, shape `(C_in, H, W)`.
- `weight`: floats, shape `(C_out, C_in, kH, kW)`, with `kH ≥ 1` and `kW ≥ 1`.
- `bias`: floats, shape `(C_out,)`.
- `stride`: an integer `≥ 1`, shared by both axes.
- `padding`: an integer `≥ 0`. That many zeros are added on the top, bottom, left and right. Channels are not padded.

Return a float array of shape `(C_out, H_out, W_out)`, where

```
H_out = (H + 2 * padding - kH) // stride + 1
W_out = (W + 2 * padding - kW) // stride + 1
```

`//` is floor division. Raise `ValueError` when `H_out < 1` or `W_out < 1`, when `stride < 1` or `padding < 0`, when the ranks or channel counts disagree, or when `bias` does not have one entry per output channel.

At output position `(c, i, j)` the window starts at `(i * stride, j * stride)` in the padded image:

```
y[c, i, j] = bias[c] + Σ_{ic, u, v} x_pad[ic, i*stride + u, j*stride + v] * weight[c, ic, u, v]
```

`u` runs over `0 .. kH-1` and `v` over `0 .. kW-1`. A mathematical convolution would multiply by `weight[c, ic, kH-1-u, kW-1-v]` instead.

```
solve([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[[1, 2], [3, 4]]]], [0.5], 1, 0)
-> array([[[37.5, 47.5], [67.5, 77.5]]])
```

The top-left value is `0.5 + 1·1 + 2·2 + 4·3 + 5·4 = 37.5`. The filter rotated 180° would produce 23.5 there.

**Interview focus:** the output-shape formula, stride and padding, and cross-correlation versus a flipped convolution.

**Follow-ups** to answer out loud once your solution passes:

- How many multiply-adds does a 3×3 convolution with 64 input and 128 output channels do on a 56×56 image?
- How would you write it as a matrix product (im2col), and what does that cost in memory?
- What do dilation and groups change? What is a depthwise convolution?
