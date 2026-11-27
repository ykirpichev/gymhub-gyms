# Occupancy calculator

Write `solve(threads_per_block, regs_per_thread, smem_per_block, sm)` that returns how many blocks of a kernel fit on one SM, the occupancy, and which resource limits it.

`sm` is a dict with integer limits:

- `"max_threads"`: resident threads per SM (for example 2048)
- `"max_blocks"`: resident blocks per SM (for example 32)
- `"registers"`: 32-bit registers per SM (for example 65536)
- `"shared"`: bytes of shared memory per SM for blocks (for example 102400)
- `"max_threads_per_block"`: the largest block allowed (for example 1024)

Rules:

1. Threads are allocated in whole warps of 32: a block of 100 threads takes 4 warps, 128 thread slots and 128 threads' worth of registers.
2. Blocks per SM is the smallest of four limits: by threads (`max_threads // threads_alloc`), by blocks (`max_blocks`), by registers (`registers // (regs_per_thread * threads_alloc)`) and by shared memory (`shared // smem_per_block`). A resource the kernel does not use (0 registers or 0 bytes) does not limit it.
3. Occupancy is resident warps over the SM's maximum warps: `blocks * warps_per_block / (max_threads / 32)`, a float.
4. The limiter is the name of the limit that equals the minimum: `"threads"`, `"blocks"`, `"registers"` or `"shared"`. On a tie, the first in that order wins.

Raise `ValueError` if `threads_per_block < 1`, if it exceeds `max_threads_per_block`, or if a single block does not fit (needs more registers or shared memory than the SM has).

Return a tuple `(blocks, occupancy, limiter)`.

```
sm = {"max_threads": 2048, "max_blocks": 32, "registers": 65536, "shared": 102400, "max_threads_per_block": 1024}
solve(256, 64, 0, sm)  ->  (4, 0.5, "registers")
```

#### What interviewers look for

- Rounding threads up to whole warps before anything else.
- All four limits, not just the one the question hints at.
- Saying what you would change to raise occupancy, and whether you should.

#### Follow-ups

- Real GPUs allocate registers per warp in chunks (for example 256 registers) and shared memory in chunks too. How would rule 2 change?
- The kernel uses 72 registers per thread. Which `__launch_bounds__` would you try, and what is the risk?
- Your kernel is at 25% occupancy and at 85% of memory bandwidth. Do you work on occupancy?
