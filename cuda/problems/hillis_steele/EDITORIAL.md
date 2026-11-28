#### Idea
After the step with distance `d`, element `i` holds the sum of the `2d` elements ending at `i` (or of all elements up
to `i`). Doubling `d` each step finishes in `⌈log₂ n⌉` steps.

#### Work versus steps
The additions add up to about `n log₂ n`, against `n − 1` for a sequential loop: Hillis–Steele trades work for few
steps. Inside a warp that is the right trade (the lanes are there anyway, and `__shfl_up_sync` makes each step one
instruction). For long arrays, work-efficient Blelloch or CUB's single-pass decoupled look-back wins.

#### Pitfalls
- Updating in place from left to right: `x[i - d]` would already include this step's addition. Use a copy (in CUDA:
  double-buffered shared memory, or read into a register, `__syncthreads()`, then write).
