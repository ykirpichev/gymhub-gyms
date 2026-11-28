# Lesson: SMs, warps and latency hiding

A GPU is a set of **streaming multiprocessors** (SMs). When you launch a grid, the hardware hands whole blocks to SMs
as room frees up; a block stays on its SM until all its threads finish. Inside the SM, threads run in groups of 32
called **warps**: threads 0–31 of a block are warp 0, 32–63 warp 1, and so on.

<figure class="diagram"><svg viewBox="0 0 660 270" width="660" role="img" aria-label="One SM: four sub-partitions, each with a warp scheduler, registers, CUDA cores and a Tensor Core, sharing L1 and shared memory">
<rect x="10" y="10" width="640" height="250" rx="10" class="d-fill"/>
<text x="330" y="30" class="d-text d-bold" text-anchor="middle">One SM (Hopper H100: 132 of them on the chip)</text>
<rect x="24" y="42" width="146" height="160" rx="6" class="d-box"/>
<rect x="180" y="42" width="146" height="160" rx="6" class="d-box"/>
<rect x="336" y="42" width="146" height="160" rx="6" class="d-box"/>
<rect x="492" y="42" width="146" height="160" rx="6" class="d-box"/>
<rect x="32" y="50" width="130" height="26" rx="4" class="d-a"/><text x="97" y="67" class="d-muted" text-anchor="middle">warp scheduler</text>
<rect x="188" y="50" width="130" height="26" rx="4" class="d-a"/><text x="253" y="67" class="d-muted" text-anchor="middle">warp scheduler</text>
<rect x="344" y="50" width="130" height="26" rx="4" class="d-a"/><text x="409" y="67" class="d-muted" text-anchor="middle">warp scheduler</text>
<rect x="500" y="50" width="130" height="26" rx="4" class="d-a"/><text x="565" y="67" class="d-muted" text-anchor="middle">warp scheduler</text>
<rect x="32" y="82" width="130" height="30" rx="4" class="d-c"/><text x="97" y="101" class="d-muted" text-anchor="middle">registers 64 KB</text>
<rect x="188" y="82" width="130" height="30" rx="4" class="d-c"/><text x="253" y="101" class="d-muted" text-anchor="middle">registers 64 KB</text>
<rect x="344" y="82" width="130" height="30" rx="4" class="d-c"/><text x="409" y="101" class="d-muted" text-anchor="middle">registers 64 KB</text>
<rect x="500" y="82" width="130" height="30" rx="4" class="d-c"/><text x="565" y="101" class="d-muted" text-anchor="middle">registers 64 KB</text>
<rect x="32" y="118" width="130" height="38" rx="4" class="d-b"/><text x="97" y="134" class="d-muted" text-anchor="middle">32 FP32 lanes,</text><text x="97" y="148" class="d-muted" text-anchor="middle">INT32, FP64, SFU</text>
<rect x="188" y="118" width="130" height="38" rx="4" class="d-b"/><text x="253" y="134" class="d-muted" text-anchor="middle">32 FP32 lanes,</text><text x="253" y="148" class="d-muted" text-anchor="middle">INT32, FP64, SFU</text>
<rect x="344" y="118" width="130" height="38" rx="4" class="d-b"/><text x="409" y="134" class="d-muted" text-anchor="middle">32 FP32 lanes,</text><text x="409" y="148" class="d-muted" text-anchor="middle">INT32, FP64, SFU</text>
<rect x="500" y="118" width="130" height="38" rx="4" class="d-b"/><text x="565" y="134" class="d-muted" text-anchor="middle">32 FP32 lanes,</text><text x="565" y="148" class="d-muted" text-anchor="middle">INT32, FP64, SFU</text>
<rect x="32" y="162" width="130" height="30" rx="4" class="d-a"/><text x="97" y="181" class="d-muted" text-anchor="middle">Tensor Core</text>
<rect x="188" y="162" width="130" height="30" rx="4" class="d-a"/><text x="253" y="181" class="d-muted" text-anchor="middle">Tensor Core</text>
<rect x="344" y="162" width="130" height="30" rx="4" class="d-a"/><text x="409" y="181" class="d-muted" text-anchor="middle">Tensor Core</text>
<rect x="500" y="162" width="130" height="30" rx="4" class="d-a"/><text x="565" y="181" class="d-muted" text-anchor="middle">Tensor Core</text>
<rect x="24" y="212" width="614" height="36" rx="6" class="d-c"/>
<text x="331" y="235" class="d-muted" text-anchor="middle">256 KB L1 data cache + shared memory (up to 228 KB of it as shared memory), TMA unit</text>
</svg>
<figcaption>An SM has four sub-partitions. Each warp lives in one of them; every cycle each scheduler picks one warp whose next instruction is ready and issues it for all 32 lanes.</figcaption></figure>

### SIMT: one instruction, 32 lanes

A warp issues one instruction at a time for all its 32 threads (single instruction, multiple threads). When threads of
a warp take different sides of a branch, the warp runs each side in turn with the other lanes masked off: that is
**divergence**. A branch that every lane of a warp takes the same way costs nothing extra.

```widget
divergence cond=even then=4 else=4
```

Since Volta (2017), each thread has its own program counter (**independent thread scheduling**), so diverged lanes may
interleave and even wait on each other without deadlock. Code that silently assumed a warp runs in lockstep (the old
"warp-synchronous" tricks) must now say so explicitly with `__syncwarp()` or the `*_sync` shuffle and vote intrinsics.

### Latency hiding and occupancy

A load from global memory takes around 500 cycles or more. The SM does not wait: while one warp is stalled, the
schedulers issue instructions from other resident warps. Switching costs nothing because every resident warp keeps its
registers in the register file all the time. The more warps that are resident, the more chances to find a ready one.

**Occupancy** is the number of resident warps divided by the SM's maximum (64 on H100 and B200, 48 on consumer cards).
It is limited by whichever runs out first: warp slots, block slots, registers (64 K 32-bit registers per SM) or shared
memory. Try it:

```widget
occupancy arch=h100 threads=256 regs=64 smem=48
```

> [!NOTE]
> High occupancy is a means, not a goal. A kernel that keeps many loads in flight per thread (instruction-level
> parallelism, e.g. each thread processing 4 elements) can saturate memory at 25–50% occupancy, and GEMM kernels
> deliberately run with few warps and many registers. Raise occupancy when the profiler says warps are stalled with no
> one ready to issue.

### Blocks and synchronization

- Threads of a block can share **shared memory** and wait for each other at `__syncthreads()`.
- Every thread of the block must reach the same `__syncthreads()`. Calling it inside a branch that only some threads
  take is undefined behaviour and usually hangs.
- Blocks cannot wait for each other inside a kernel (they may not even be resident at the same time). To synchronize
  the whole grid, end the kernel and launch another, or use a cooperative launch. Hopper adds **thread block
  clusters**, groups of blocks guaranteed to run together; the architecture track covers them.

### Where to go next
- In the GPU gym, [ReLU](#/gym/gpu/relu) and [reverse](#/gym/gpu/reverse) are short kernels to try divergence and
  indexing on real hardware.
- The GPU interview course has more drills on [divergence](#/gym/gpu-interview/warp_divergence) and
  [occupancy](#/gym/gpu-interview/occupancy_calculator).
