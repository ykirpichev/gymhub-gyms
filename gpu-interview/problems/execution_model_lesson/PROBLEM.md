# The CUDA execution model in one page

A kernel launch `kernel<<<grid, block>>>(args)` starts `grid × block` threads, all running the same function. Each thread finds its work from built-in variables:

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;   // global index in a 1D launch
if (i < n) y[i] = a * x[i] + y[i];               // the bounds check: the grid is rounded up
```

<figure class="diagram"><svg viewBox="0 0 680 232" width="680" role="img" aria-label="A grid of blocks, a block of warps, a warp of 32 lanes">
<rect x="10" y="30" width="190" height="150" rx="8" class="d-fill"/>
<text x="20" y="50" class="d-text d-bold" text-anchor="start">Grid: 6 blocks</text>
<rect x="22" y="62" width="50" height="48" rx="4" class="d-a"/>
<text x="47" y="91" class="d-muted" text-anchor="middle">block 0</text>
<rect x="82" y="62" width="50" height="48" rx="4" class="d-box"/>
<text x="107" y="91" class="d-muted" text-anchor="middle">block 1</text>
<rect x="142" y="62" width="50" height="48" rx="4" class="d-box"/>
<text x="167" y="91" class="d-muted" text-anchor="middle">block 2</text>
<rect x="22" y="118" width="50" height="48" rx="4" class="d-box"/>
<text x="47" y="147" class="d-muted" text-anchor="middle">block 3</text>
<rect x="82" y="118" width="50" height="48" rx="4" class="d-box"/>
<text x="107" y="147" class="d-muted" text-anchor="middle">block 4</text>
<rect x="142" y="118" width="50" height="48" rx="4" class="d-box"/>
<text x="167" y="147" class="d-muted" text-anchor="middle">block 5</text>
<rect x="240" y="30" width="200" height="150" rx="8" class="d-a"/>
<text x="250" y="50" class="d-text d-bold" text-anchor="start">Block 0: 128 threads</text>
<rect x="252" y="62" width="176" height="22" rx="3" class="d-b"/>
<text x="262" y="77" class="d-muted" text-anchor="start">warp 0 · threads 0–31</text>
<rect x="252" y="90" width="176" height="22" rx="3" class="d-box"/>
<text x="262" y="105" class="d-muted" text-anchor="start">warp 1 · threads 32–63</text>
<rect x="252" y="118" width="176" height="22" rx="3" class="d-box"/>
<text x="262" y="133" class="d-muted" text-anchor="start">warp 2 · threads 64–95</text>
<rect x="252" y="146" width="176" height="22" rx="3" class="d-box"/>
<text x="262" y="161" class="d-muted" text-anchor="start">warp 3 · threads 96–127</text>
<rect x="480" y="30" width="190" height="150" rx="8" class="d-b"/>
<text x="490" y="50" class="d-text d-bold" text-anchor="start">Warp 0: 32 lanes</text>
<rect x="494" y="62" width="17" height="17" rx="2" class="d-box"/>
<rect x="515" y="62" width="17" height="17" rx="2" class="d-box"/>
<rect x="536" y="62" width="17" height="17" rx="2" class="d-box"/>
<rect x="557" y="62" width="17" height="17" rx="2" class="d-box"/>
<rect x="578" y="62" width="17" height="17" rx="2" class="d-box"/>
<rect x="599" y="62" width="17" height="17" rx="2" class="d-box"/>
<rect x="620" y="62" width="17" height="17" rx="2" class="d-box"/>
<rect x="641" y="62" width="17" height="17" rx="2" class="d-box"/>
<rect x="494" y="84" width="17" height="17" rx="2" class="d-box"/>
<rect x="515" y="84" width="17" height="17" rx="2" class="d-box"/>
<rect x="536" y="84" width="17" height="17" rx="2" class="d-box"/>
<rect x="557" y="84" width="17" height="17" rx="2" class="d-box"/>
<rect x="578" y="84" width="17" height="17" rx="2" class="d-box"/>
<rect x="599" y="84" width="17" height="17" rx="2" class="d-box"/>
<rect x="620" y="84" width="17" height="17" rx="2" class="d-box"/>
<rect x="641" y="84" width="17" height="17" rx="2" class="d-box"/>
<rect x="494" y="106" width="17" height="17" rx="2" class="d-box"/>
<rect x="515" y="106" width="17" height="17" rx="2" class="d-box"/>
<rect x="536" y="106" width="17" height="17" rx="2" class="d-box"/>
<rect x="557" y="106" width="17" height="17" rx="2" class="d-box"/>
<rect x="578" y="106" width="17" height="17" rx="2" class="d-box"/>
<rect x="599" y="106" width="17" height="17" rx="2" class="d-box"/>
<rect x="620" y="106" width="17" height="17" rx="2" class="d-box"/>
<rect x="641" y="106" width="17" height="17" rx="2" class="d-box"/>
<rect x="494" y="128" width="17" height="17" rx="2" class="d-box"/>
<rect x="515" y="128" width="17" height="17" rx="2" class="d-box"/>
<rect x="536" y="128" width="17" height="17" rx="2" class="d-box"/>
<rect x="557" y="128" width="17" height="17" rx="2" class="d-box"/>
<rect x="578" y="128" width="17" height="17" rx="2" class="d-box"/>
<rect x="599" y="128" width="17" height="17" rx="2" class="d-box"/>
<rect x="620" y="128" width="17" height="17" rx="2" class="d-box"/>
<rect x="641" y="128" width="17" height="17" rx="2" class="d-box"/>
<text x="575" y="168" class="d-muted" text-anchor="middle">one instruction for all lanes</text>
<line x1="202" y1="86" x2="236" y2="86" class="d-line d-dash" marker-end="url(#d-arrow)"/>
<line x1="428" y1="73" x2="476" y2="100" class="d-line d-dash" marker-end="url(#d-arrow)"/>
<text x="105" y="204" class="d-muted" text-anchor="middle">Blocks go to SMs in any order;</text>
<text x="105" y="220" class="d-muted" text-anchor="middle">they cannot sync with each other</text>
<text x="340" y="204" class="d-muted" text-anchor="middle">Threads share shared memory</text>
<text x="340" y="220" class="d-muted" text-anchor="middle">and meet at __syncthreads()</text>
<text x="575" y="204" class="d-muted" text-anchor="middle">SIMT: lanes that branch apart</text>
<text x="575" y="220" class="d-muted" text-anchor="middle">run both sides in turn</text>
</svg>
<figcaption>What one launch <code>kernel&lt;&lt;&lt;6, 128&gt;&gt;&gt;</code> creates. The grid is split into blocks, each block into warps of 32 threads, and a warp runs one instruction at a time for all its lanes.</figcaption></figure>

**Blocks** are scheduled onto **streaming multiprocessors (SMs)**. A block stays on one SM until it finishes, so its threads can share that SM's **shared memory** and synchronize with `__syncthreads()`. Blocks cannot synchronize with each other inside a kernel (short of cooperative launches), which is why a global reduction needs a second kernel or atomics.

Inside a block, threads run in groups of 32 called **warps**. A warp issues one instruction at a time for all its active lanes (*SIMT*). When lanes of a warp take different sides of a branch, the warp runs both sides one after the other with some lanes masked off: **divergence**. Different warps diverging costs nothing extra.

An SM keeps many warps resident and switches between them every cycle to hide latency: while one warp waits about 500 cycles for global memory, others compute. How many warps fit is **occupancy**, limited by registers per thread, shared memory per block, and thread and block limits.

<figure class="diagram"><svg viewBox="0 0 680 214" width="680" role="img" aria-label="GPU memory hierarchy: registers, shared memory, L2, HBM">
<rect x="190.0" y="18" width="140" height="36" rx="6" class="d-a"/>
<text x="260" y="34" class="d-text d-bold" text-anchor="middle">Registers</text>
<text x="260" y="48" class="d-muted" text-anchor="middle">per thread</text>
<text x="520" y="34" class="d-text" text-anchor="start">~1 cycle</text>
<text x="520" y="48" class="d-muted" text-anchor="start">256 KB per SM</text>
<rect x="135.0" y="64" width="250" height="36" rx="6" class="d-b"/>
<text x="260" y="80" class="d-text d-bold" text-anchor="middle">Shared memory / L1</text>
<text x="260" y="94" class="d-muted" text-anchor="middle">per SM (per block)</text>
<text x="520" y="80" class="d-text" text-anchor="start">~30 cycles</text>
<text x="520" y="94" class="d-muted" text-anchor="start">up to 228 KB per SM</text>
<rect x="80.0" y="110" width="360" height="36" rx="6" class="d-c"/>
<text x="260" y="126" class="d-text d-bold" text-anchor="middle">L2 cache</text>
<text x="260" y="140" class="d-muted" text-anchor="middle">whole GPU</text>
<text x="520" y="126" class="d-text" text-anchor="start">~200 cycles</text>
<text x="520" y="140" class="d-muted" text-anchor="start">~50 MB</text>
<rect x="25.0" y="156" width="470" height="36" rx="6" class="d-box"/>
<text x="260" y="172" class="d-text d-bold" text-anchor="middle">HBM (global memory)</text>
<text x="260" y="186" class="d-muted" text-anchor="middle">whole GPU</text>
<text x="520" y="172" class="d-text" text-anchor="start">~500 cycles</text>
<text x="520" y="186" class="d-muted" text-anchor="start">80 GB at ~3 TB/s</text>
<line x1="505" y1="22" x2="505" y2="188" class="d-line" marker-end="url(#d-arrow)"/>
<text x="498" y="205" class="d-muted" text-anchor="end">bigger, slower</text>
</svg>
<figcaption>The memory hierarchy of an H100-class GPU, rounded. Each step down is roughly ten times larger and several times slower. Kernels get fast by keeping reused data in the top two rows.</figcaption></figure>

| Memory | Scope | Rough latency | Notes |
|---|---|---|---|
| Registers | one thread | ~1 cycle | 64K 32-bit registers per SM, shared by all resident threads |
| Shared memory | one block | ~20–30 cycles | on-chip, 32 banks, managed by you |
| L1 / L2 cache | SM / whole GPU | ~30 / ~200 cycles | hardware-managed |
| Global memory (HBM) | all threads, the host | ~400–800 cycles | terabytes per second, but only with coalesced access |
| Local memory | one thread | as global | register spills and dynamically indexed arrays; lives in global memory |

> [!NOTE]
> Latency is hidden by parallelism, bandwidth is not. A kernel with enough warps in flight stops caring about latency and starts caring about bytes moved, which is why the next tracks count bytes.
