# Shuffle-down reduction

Simulate `__shfl_down_sync` reductions. `lanes` is a list of 32 numbers, one per lane. Run the loop
`for offset in 16, 8, 4, 2, 1: v[i] += shfl_down(v, offset)[i]` where `shfl_down(v, offset)[i]` is `v[i + offset]` if
`i + offset < 32`, and `v[i]` itself otherwise (a lane with no source reads its own value back, like the hardware).
All lanes update at the same time from the values before the step.

Return the 32 values after all five steps. Lane 0 holds the sum; the other lanes hold partial (and some doubled) sums.

Example: with every lane holding 1, every lane ends with 32 (lane 0 has summed all 32 lanes; the upper lanes
read their own value back and double it). With `lanes = [0, 1, 2, …, 31]`, lane 0 ends with 496, the sum.
