# Lesson: where data lives, and coalescing

Most kernels are limited by memory, not arithmetic. Knowing where each variable lives, and how a warp's loads turn
into memory transactions, is the most useful performance skill in CUDA.

<figure class="diagram"><svg viewBox="0 0 660 250" width="660" role="img" aria-label="Memory hierarchy from registers to HBM with sizes and latencies">
<rect x="250" y="12" width="160" height="34" rx="6" class="d-b"/>
<text x="330" y="34" class="d-text" text-anchor="middle">registers</text>
<rect x="200" y="56" width="260" height="34" rx="6" class="d-b"/>
<text x="330" y="78" class="d-text" text-anchor="middle">shared memory / L1 (per SM)</text>
<rect x="140" y="100" width="380" height="34" rx="6" class="d-a"/>
<text x="330" y="122" class="d-text" text-anchor="middle">L2 cache (whole chip)</text>
<rect x="70" y="144" width="520" height="34" rx="6" class="d-c"/>
<text x="330" y="166" class="d-text" text-anchor="middle">HBM / GDDR global memory (device DRAM)</text>
<rect x="10" y="188" width="640" height="34" rx="6" class="d-fill"/>
<text x="330" y="210" class="d-text" text-anchor="middle">host memory over PCIe or NVLink-C2C</text>
<text x="420" y="34" class="d-muted" text-anchor="start">~1 cycle · 256 KB per SM</text>
<text x="470" y="78" class="d-muted" text-anchor="start">~30 cycles · up to 228 KB</text>
<text x="530" y="122" class="d-muted" text-anchor="start">~200–300 cycles · 50–126 MB</text>
<text x="20" y="166" class="d-muted" text-anchor="start">~500+ cycles</text>
<text x="600" y="166" class="d-muted" text-anchor="end">3–8 TB/s · 80–192 GB</text>
<text x="330" y="242" class="d-muted" text-anchor="middle">~1–5 µs per transfer · 25–64 GB/s over PCIe 4/5, ~450 GB/s per direction over C2C</text>
</svg>
<figcaption>Rough numbers for a Hopper or Blackwell data-center GPU. Each level down is bigger, slower and shared by more threads.</figcaption></figure>

| Space | Declared as | Scope | Lives in |
|---|---|---|---|
| Registers | local scalars | one thread | register file |
| Local memory | big or indexed local arrays, spills | one thread | DRAM, cached in L1/L2 |
| Shared memory | `__shared__` | one block (or a cluster on Hopper+) | on-chip SRAM, same as L1 |
| Global memory | `cudaMalloc`, `__device__` | all threads, all kernels | DRAM, cached in L2 (and L1) |
| Constant memory | `__constant__` | all threads, read-only | DRAM, with a broadcast cache |

> [!WARNING]
> "Local" memory is not local in speed: it is off-chip DRAM. A local array indexed by a runtime value
> (`float acc[8]; acc[k] += …` with unknown `k`) or a kernel with too many registers spills there. Compile with
> `-Xptxas -v` to see registers and spill bytes per kernel.

### Coalescing: how a warp's loads become transactions

When a warp executes a load, the 32 addresses are combined. Memory moves in **32-byte sectors** (four of them make a
128-byte cache line). If the 32 lanes read 32 consecutive floats, that is 128 bytes: exactly four sectors, no waste.
If each lane reads every second float, the warp still drags in eight sectors and uses half of each. Explore it:

```widget
coalescing bytes=4 stride=1 offset=0
```

Try a stride of 2, then 32 (each lane touches its own sector, 12.5% efficiency for floats), then switch to `float4`
with stride 1: each lane reads 16 bytes, the warp 512 bytes, still at 100%. Wider loads (`float4`, `__half2`) also mean
fewer instructions for the same bytes.

### Structures of arrays

The most common cause of strided access is an array of structures:

```cuda
struct Particle { float x, y, z, mass; };
Particle* p;               // AoS: p[i].x is 16 bytes away from p[i+1].x → stride 4 floats
float *x, *y, *z, *mass;   // SoA: x[i] and x[i+1] are neighbours → coalesced
```

If a kernel reads only `x`, the AoS layout wastes 3/4 of every sector; the SoA layout wastes nothing.

### Getting data onto the GPU

- `cudaMalloc` plus `cudaMemcpy` is explicit. Copies from **pinned** host memory (`cudaMallocHost`) run at full PCIe
  speed and can be asynchronous; copies from ordinary pageable memory are staged through a pinned buffer first.
- **Unified memory** (`cudaMallocManaged`) gives one pointer valid on both sides; pages migrate on demand when touched.
  It is convenient, but first-touch page faults can make a kernel much slower than with explicit copies; prefetch with
  `cudaMemPrefetchAsync`. On Grace Hopper and Grace Blackwell (and Jetson, where CPU and GPU share the same DRAM) the
  GPU can read host memory directly over a coherent link.
- Keep data on the device between kernels. A PCIe copy of a tensor usually costs more than the kernel that uses it.

### Where to go next
- In the GPU gym, [transpose](#/gym/gpu/transpose) shows coalesced versus strided access on a real GPU, and
  [matvec](#/gym/gpu/matvec) shows why the layout of the matrix decides which threads read what.
- The roofline (in the performance track) turns bytes moved into a time you can predict.
