#### Idea
Each step halves the number of lanes that still carry a useful partial: after offset 16, lanes 0–15 hold pairs; after
8, lanes 0–7 hold groups of four; after offset 1, lane 0 holds all 32. The upper lanes compute garbage, which costs
nothing: the whole warp executes each shuffle anyway.

#### Pitfalls
- Updating in place: lane `i` would read a value that lane `i + offset` already updated this step. Shuffles read the
  registers as they were before the instruction.
- Expecting every lane to hold the sum. For that, use `__shfl_xor_sync` (a butterfly) or broadcast lane 0's value with
  `__shfl_sync(mask, v, 0)`.
- Using a partial mask: the `0xffffffff` mask must name every lane that participates, and they must all execute the call.
