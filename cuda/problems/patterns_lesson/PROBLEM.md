# Lesson: reduce, scan, and the warp-level toolbox

Many kernels combine values rather than map them: a sum, a max, a prefix sum, a histogram. On a GPU these become
**trees**, so that `n` values combine in `log₂ n` steps instead of `n`.

### Reduction in shared memory

Each block loads its values into shared memory and halves the active threads each step. Step through it, and compare
the two addressing schemes:

```widget
reduction n=16 mode=interleaved warp=4
```

*Interleaved* addressing (`if (t % (2*s) == 0)`) leaves a few active lanes in every warp, so all warps stay busy and the
modulo is slow. *Sequential* addressing (`if (t < s)`) keeps active threads contiguous: after a few steps whole warps
retire, and the accesses are conflict-free. Mark Harris's classic talk takes this from 2 GB/s to over 60 GB/s on an
old GPU in seven steps; the same steps still apply.

### Warp shuffles

Inside a warp, lanes can read each other's registers directly, with no shared memory and no `__syncthreads()`:

```cuda
__inline__ __device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset /= 2)
        v += __shfl_down_sync(0xffffffff, v, offset);   // lane i reads lane i + offset
    return v;                                           // lane 0 holds the warp's sum
}
```

<figure class="diagram"><svg viewBox="0 0 640 200" width="640" role="img" aria-label="A shuffle-down reduction over 8 lanes in three steps">
<text x="20" y="34" class="d-muted" text-anchor="start">lanes</text>
<rect x="80" y="20" width="56" height="24" rx="4" class="d-a"/><text x="108" y="36" class="d-muted" text-anchor="middle">0</text>
<rect x="150" y="20" width="56" height="24" rx="4" class="d-box"/><text x="178" y="36" class="d-muted" text-anchor="middle">1</text>
<rect x="220" y="20" width="56" height="24" rx="4" class="d-box"/><text x="248" y="36" class="d-muted" text-anchor="middle">2</text>
<rect x="290" y="20" width="56" height="24" rx="4" class="d-box"/><text x="318" y="36" class="d-muted" text-anchor="middle">3</text>
<rect x="360" y="20" width="56" height="24" rx="4" class="d-box"/><text x="388" y="36" class="d-muted" text-anchor="middle">4</text>
<rect x="430" y="20" width="56" height="24" rx="4" class="d-box"/><text x="458" y="36" class="d-muted" text-anchor="middle">5</text>
<rect x="500" y="20" width="56" height="24" rx="4" class="d-box"/><text x="528" y="36" class="d-muted" text-anchor="middle">6</text>
<rect x="570" y="20" width="56" height="24" rx="4" class="d-box"/><text x="598" y="36" class="d-muted" text-anchor="middle">7</text>
<text x="20" y="86" class="d-muted" text-anchor="start">offset 4</text>
<line x1="388" y1="46" x2="112" y2="70" class="d-line" marker-end="url(#d-arrow)"/>
<line x1="458" y1="46" x2="182" y2="70" class="d-line" marker-end="url(#d-arrow)"/>
<line x1="528" y1="46" x2="252" y2="70" class="d-line" marker-end="url(#d-arrow)"/>
<line x1="598" y1="46" x2="322" y2="70" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="80" y="72" width="56" height="24" rx="4" class="d-a"/><rect x="150" y="72" width="56" height="24" rx="4" class="d-b"/>
<rect x="220" y="72" width="56" height="24" rx="4" class="d-b"/><rect x="290" y="72" width="56" height="24" rx="4" class="d-b"/>
<text x="20" y="138" class="d-muted" text-anchor="start">offset 2</text>
<line x1="248" y1="98" x2="112" y2="122" class="d-line" marker-end="url(#d-arrow)"/>
<line x1="318" y1="98" x2="182" y2="122" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="80" y="124" width="56" height="24" rx="4" class="d-a"/><rect x="150" y="124" width="56" height="24" rx="4" class="d-b"/>
<text x="20" y="190" class="d-muted" text-anchor="start">offset 1</text>
<line x1="178" y1="150" x2="112" y2="174" class="d-line" marker-end="url(#d-arrow)"/>
<rect x="80" y="176" width="56" height="24" rx="4" class="d-c"/><text x="108" y="192" class="d-muted" text-anchor="middle">sum</text>
<text x="420" y="140" class="d-muted" text-anchor="start">every lane executes each step;</text>
<text x="420" y="156" class="d-muted" text-anchor="start">only the lanes on the left</text>
<text x="420" y="172" class="d-muted" text-anchor="start">keep a meaningful value</text>
</svg>
<figcaption>Shuffle-down with offsets 4, 2, 1 over 8 lanes (a real warp uses 16, 8, 4, 2, 1). Use __shfl_xor_sync instead and every lane ends with the total: a butterfly.</figcaption></figure>

A block reduction is then: every warp reduces with shuffles, lane 0 of each warp writes its partial to shared memory,
one warp reduces the partials. Other warp-level intrinsics: `__ballot_sync` (a 32-bit mask of a predicate),
`__any_sync`/`__all_sync`, `__match_any_sync`, and on Ampere+ `__reduce_add_sync` for integers. **Cooperative groups**
(`cg::tiled_partition<32>(block)`, `cg::reduce`) wrap all of this in a typed API.

### Scan (prefix sum)

An inclusive scan turns `[3, 1, 7, 0]` into `[3, 4, 11, 11]`. It looks sequential but parallelizes:

- **Hillis–Steele**: at step `d`, every element adds the one `2^d` places to its left. `log₂ n` steps, `n log n` work;
  simple and fast for one warp (with `__shfl_up_sync`).
- **Blelloch**: an up-sweep builds a sum tree, a down-sweep distributes prefixes. `2 log₂ n` steps but only `O(n)`
  work, better for large blocks.
- Large arrays: scan each block, scan the block totals, add them back. Modern libraries (CUB's
  `DeviceScan`) do it in a single pass with **decoupled look-back**, at the speed of a copy.

Scans power stream compaction (keep the elements that pass a test), radix sort, sparse matrix formats and top-k.

### Atomics and histograms

`atomicAdd(&x, v)` performs the read-modify-write as one indivisible operation. Atomics on global memory are resolved in
the L2 cache and are fast when addresses are spread out, slow when thousands of threads hit the same address. A
histogram therefore **privatizes**: each block builds a histogram in shared memory with shared atomics, then adds it to
the global one with one atomic per bin.

> [!WARNING]
> Floating-point addition is not associative. A sum computed with atomics or with a different tree shape gives slightly
> different results run to run. If you need bit-for-bit reproducibility, use a fixed tree order.

### Where to go next
- GPU gym: [reduction](#/gym/gpu/reduction), [prefix sum](#/gym/gpu/prefix_sum), [histogram](#/gym/gpu/histogram),
  [row softmax](#/gym/gpu/row_softmax), [top-k](#/gym/gpu/top_k) and [sort](#/gym/gpu/sort).
- The GPU interview course has a [butterfly reduction](#/gym/gpu-interview/warp_butterfly) and a
  [Blelloch scan](#/gym/gpu-interview/blelloch_scan) to code.
