#### The idea
Round the block up to whole warps first; every other limit uses the allocated thread count. Then compute the four
limits, take the smallest, and report which one it was.

#### A solution
```python
def solve(threads_per_block, regs_per_thread, smem_per_block, sm):
    if threads_per_block < 1 or threads_per_block > sm["max_threads_per_block"]:
        raise ValueError("bad block size")
    warps = (threads_per_block + 31) // 32
    threads = warps * 32
    regs = regs_per_thread * threads
    if regs > sm["registers"] or smem_per_block > sm["shared"]:
        raise ValueError("one block does not fit on an SM")
    limits = [
        (sm["max_threads"] // threads, "threads"),
        (sm["max_blocks"], "blocks"),
        (sm["registers"] // regs if regs else float("inf"), "registers"),
        (sm["shared"] // smem_per_block if smem_per_block else float("inf"), "shared"),
    ]
    blocks = min(count for count, _ in limits)
    limiter = next(name for count, name in limits if count == blocks)   # first on a tie
    return (int(blocks), blocks * warps / (sm["max_threads"] / 32), limiter)
```

#### Reading the answer
In the example, 64 registers × 256 threads = 16,384 registers per block, so 4 blocks fit in 65,536: 32 of 64 warps,
50 percent occupancy, limited by registers. Dropping to 32 registers per thread (`__launch_bounds__` or
`-maxrregcount`) would double it, at the risk of spilling to local memory.

Higher occupancy hides latency better, but it is not the goal in itself: a matmul with large register tiles often
runs fastest at 25 to 50 percent, because each warp has more independent work in flight (instruction-level
parallelism).

#### Pitfalls
- Forgetting the warp rounding: a 100-thread block costs 128 threads' worth of registers and slots.
- Treating an unused resource (0 registers or 0 bytes of shared memory) as a limit of 0 blocks.
- Real GPUs also allocate registers and shared memory in chunks (for example 256 registers per warp); the exercise
  leaves that out, the CUDA occupancy calculator does not.
