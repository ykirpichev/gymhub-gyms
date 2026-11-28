#### Idea
Both resources work in parallel, so the time is at least the larger of the two times. The ratio `flops / bytes` against
`peak / bandwidth` (the ridge point) says which one wins.

#### Use it
Before optimizing, compute this bound and compare it with the measured time. At 3× the bound there is room; at 1.1× the
kernel is done and only an algorithm that moves fewer bytes or does fewer FLOPs helps. Real kernels reach 80–90% of
bandwidth and 60–80% of Tensor Core peak.

#### Pitfalls
- Units: TFLOP/s is 10¹² FLOP/s and TB/s is 10¹² B/s (not 2⁴⁰).
- Counting a multiply-add as one FLOP: the convention is two.
