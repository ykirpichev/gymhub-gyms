#### Idea
A floating-point format is a set of evenly spaced values within each power-of-two interval `[2^e, 2^(e+1))`, with
`2^m` steps per interval for `m` mantissa bits. So: find the interval, compute the spacing, round to a multiple.
Below the smallest normal exponent the spacing stops shrinking: those are the subnormals.

#### Why it matters
FP8 keeps only about two significant decimal digits: 300 becomes 288. That is why FP8 training multiplies tensors by a
scale before the cast (so the values use the format's range well), keeps master weights and accumulations in higher
precision, and why Blackwell adds per-block scales.

#### Pitfalls
- Rounding half away from zero (`int(x + 0.5)`) instead of to even: hardware conversions round ties to even.
- Forgetting saturation: without `SATFINITE`, values above 448 become NaN in E4M3.
