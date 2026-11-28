# Lesson: shared memory, banks and tiles

**Shared memory** is a programmer-managed scratchpad on each SM, visible to all threads of a block. It is about as fast
as L1 (it *is* the same SRAM), and it lets threads cooperate: load a piece of global memory once, then let many threads
reuse it.

The pattern is always the same:

```cuda
__shared__ float tile[TILE][TILE];
tile[ty][tx] = in[...];      // 1. each thread loads one element (coalesced)
__syncthreads();             // 2. wait until the whole tile is loaded
float v = tile[tx][ty];      // 3. read anything in the tile, in any order
__syncthreads();             // 4. before the tile is overwritten in the next iteration
```

### Banks

Shared memory is split into **32 banks**; consecutive 4-byte words go to consecutive banks (word `k` is in bank
`k % 32`). In one cycle each bank delivers one word. If several lanes of a warp read *different* words of the same bank,
those reads are serialized: an *n-way* bank conflict costs *n* cycles. Lanes that read the *same* word get it by
broadcast, at no cost.

```widget
bank-conflicts stride=32
```

A stride of 32 words (reading down a column of a `float[32][32]` tile) puts every lane in bank 0: a 32-way conflict.
Change the stride to 33 to see the classic fix: **pad** each row by one element, `float tile[32][33]`, so that
element `[r][c]` lands in bank `(33r + c) % 32 = (r + c) % 32`, and a column spreads over all 32 banks. Odd strides
never conflict; even strides do.

### Tiled transpose

A naive transpose reads rows and writes columns: one side is always strided. Staging a 32 × 32 tile through shared
memory makes both global accesses coalesced: read a tile row-wise, write it out row-wise from the transposed position.

<figure class="diagram"><svg viewBox="0 0 640 170" width="640" role="img" aria-label="Transpose through a shared memory tile">
<rect x="10" y="30" width="150" height="110" rx="6" class="d-fill"/>
<text x="85" y="22" class="d-text" text-anchor="middle">input (global)</text>
<rect x="30" y="50" width="60" height="60" class="d-a"/>
<line x1="36" y1="60" x2="84" y2="60" class="d-line" marker-end="url(#d-arrow)"/>
<line x1="36" y1="74" x2="84" y2="74" class="d-line" marker-end="url(#d-arrow)"/>
<line x1="36" y1="88" x2="84" y2="88" class="d-line" marker-end="url(#d-arrow)"/>
<text x="85" y="132" class="d-muted" text-anchor="middle">read rows: coalesced</text>
<rect x="245" y="40" width="150" height="100" rx="6" class="d-b"/>
<text x="320" y="22" class="d-text" text-anchor="middle">shared tile [32][33]</text>
<text x="320" y="85" class="d-muted" text-anchor="middle">store tile[ty][tx]</text>
<text x="320" y="101" class="d-muted" text-anchor="middle">read tile[tx][ty]</text>
<text x="320" y="117" class="d-muted" text-anchor="middle">(+1 pad: no conflicts)</text>
<rect x="480" y="30" width="150" height="110" rx="6" class="d-fill"/>
<text x="555" y="22" class="d-text" text-anchor="middle">output (global)</text>
<rect x="540" y="50" width="60" height="60" class="d-a"/>
<line x1="546" y1="60" x2="594" y2="60" class="d-line" marker-end="url(#d-arrow)"/>
<line x1="546" y1="74" x2="594" y2="74" class="d-line" marker-end="url(#d-arrow)"/>
<line x1="546" y1="88" x2="594" y2="88" class="d-line" marker-end="url(#d-arrow)"/>
<text x="555" y="132" class="d-muted" text-anchor="middle">write rows: coalesced</text>
<line x1="162" y1="85" x2="240" y2="85" class="d-line" marker-end="url(#d-arrow)"/>
<line x1="397" y1="85" x2="476" y2="85" class="d-line" marker-end="url(#d-arrow)"/>
<text x="320" y="160" class="d-muted" text-anchor="middle">the tile from block (bx, by) is written to block (by, bx) of the output</text>
</svg>
<figcaption>The transpose happens inside shared memory, where strided access is cheap; global memory sees only row-wise traffic.</figcaption></figure>

### Tiled matrix multiply

In `C = A × B` with N × N matrices, each element of A is used by N different outputs. A naive kernel (one thread per
`C[i][j]`, looping over `k`) loads each element from global memory N times. A **tiled** kernel lets a block compute a
T × T tile of C: it walks along K, at each step loading a T × T tile of A and one of B into shared memory, and every
thread does T multiply-adds from shared memory. Each global element is now loaded N / T times instead of N.

```widget
tiling n=12 tile=32 bytes=2
```

```cuda
for (int k0 = 0; k0 < N; k0 += T) {
    As[ty][tx] = A[row * N + k0 + tx];
    Bs[ty][tx] = B[(k0 + ty) * N + col];
    __syncthreads();
    for (int k = 0; k < T; k++) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
}
C[row * N + col] = acc;
```

Fast GEMMs go further. **Register tiling**: each thread computes a small block of C (say 8 × 8) in registers, so each
value read from shared memory is used 8 times. **Vectorized loads** (`float4`). **Double buffering**: load the next
tile while computing on this one (`cp.async` on Ampere, TMA on Hopper). And finally **Tensor Cores**, which the
architecture track covers.

### Where to go next
- GPU gym: [transpose](#/gym/gpu/transpose), [matmul](#/gym/gpu/matmul), [conv1d](#/gym/gpu/conv1d) and
  [conv2d](#/gym/gpu/conv2d) all reward shared-memory tiling.
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) walks the same path in ten kernels.
