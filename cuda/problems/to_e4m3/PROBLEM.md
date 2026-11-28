# Round to FP8 E4M3

Convert a number to the nearest FP8 E4M3 value, as a conversion with saturation does (`__nv_cvt_float_to_fp8` with
`__NV_SATFINITE`).

E4M3: exponent bias 7, 3 mantissa bits, largest finite value 448, smallest normal 2⁻⁶, subnormals down to 2⁻⁹.

Rules: zero stays zero (return `0.0`). Otherwise take `e = floor(log2(|x|))`, but not less than −6 (below that the
spacing stays that of the subnormals). The spacing of representable values is `2 ** (e - 3)`. Round `|x|` to the
nearest multiple of the spacing, ties to even (Python's `round` does ties-to-even). Clamp to 448, restore the sign.

Example: `3.14159` → `e = 1`, spacing 0.25, 12.566 spacings → 13 → `3.25`. `1000` → `448.0`. `0.001` → `e = -9`, raised
to −6, spacing 2⁻⁹ = 0.001953125: 0.512 → 1 spacing → `0.001953125`.
