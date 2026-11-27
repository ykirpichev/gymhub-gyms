## Hint 1
Channel `c` uses exponent `(c - c % 2) / d_model`: channels 0 and 1 share exponent 0, channels 2 and 3 share `2 / d_model`, and so on. A common bug is using `c / d_model`, which gives each cosine its own, wrong frequency.

## Hint 2
Build `angles = pos[:, None] / 10000 ** (np.arange(0, d_model, 2) / d_model)`, with shape `(L, ceil(d / 2))`. Fill `pe[:, 0::2]` with `sin(angles)` and `pe[:, 1::2]` with `cos` of only as many columns as there are odd channels (`d_model // 2`).
