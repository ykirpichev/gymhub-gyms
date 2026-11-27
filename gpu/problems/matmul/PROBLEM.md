# Matrix multiply

**Level:** medium   **Teaches:** data reuse, shared-memory tiling, register tiling

## Task

Multiply `A` by `B` and write the product to `C`. Each output is the dot product of a row of `A` with a column of `B`.

## Arguments

| Name | Type           | Shape   | Role   |
| ---- | -------------- | ------- | ------ |
| `A`  | `const float*` | `M × K` | input  |
| `B`  | `const float*` | `K × N` | input  |
| `C`  | `float*`       | `M × N` | output |
| `M`  | `int`          |         | size   |
| `N`  | `int`          |         | size   |
| `K`  | `int`          |         | size   |

All matrices are row-major. `A[m, k]` is at `A[m * K + k]`, `B[k, n]` is at `B[k * N + n]`, and `C[m, n]` is at `C[m * N + n]`.

## Formula

```
C[m, n] = sum over k in [0, K) of A[m, k] * B[k, n]
```

## Example

```
A = [[1, 2],     B = [[5, 6],
     [3, 4]]          [7, 8]]

C[0, 0] = 1*5 + 2*7 = 19
C[0, 1] = 1*6 + 2*8 = 22
C[1, 0] = 3*5 + 4*7 = 43
C[1, 1] = 3*6 + 4*8 = 50

C = [[19, 22],
     [43, 50]]
```

## How to think about it

Every element of `C` is independent, so one thread per output is a correct start. Each one does `K` multiply-adds.

The inputs are heavily shared. Every element of `A` is used by all `N` outputs in its row of `C`, and every element of `B` by all `M` outputs in its column. The naive kernel ignores this and loads two values from global memory for every multiply-add. For the 512 × 512 benchmark that is about 1 GB of loads, while the matrices themselves are 3 MB.

Fast kernels turn that reuse into on-chip reuse. A block loads a tile of `A` and a tile of `B` into shared memory once and every thread in the block reads them many times. Each thread then computes a small patch of `C` from values in registers. Done well, the kernel is limited by arithmetic, not memory. The benchmark reports TFLOP/s, counting `2 * M * N * K` operations.

<figure class="diagram"><svg viewBox="0 0 680 356" width="680" role="img" aria-label="Tiled matrix multiply: one C tile walks a row of A tiles and a column of B tiles">
<rect x="200" y="20" width="36" height="36" rx="0" class="d-box"/>
<rect x="236" y="20" width="36" height="36" rx="0" class="d-box"/>
<rect x="272" y="20" width="36" height="36" rx="0" class="d-b"/>
<text x="290.0" y="43" class="d-muted" text-anchor="middle">k=0</text>
<rect x="308" y="20" width="36" height="36" rx="0" class="d-box"/>
<rect x="200" y="56" width="36" height="36" rx="0" class="d-box"/>
<rect x="236" y="56" width="36" height="36" rx="0" class="d-box"/>
<rect x="272" y="56" width="36" height="36" rx="0" class="d-b"/>
<text x="290.0" y="79" class="d-muted" text-anchor="middle">k=1</text>
<rect x="308" y="56" width="36" height="36" rx="0" class="d-box"/>
<rect x="200" y="92" width="36" height="36" rx="0" class="d-box"/>
<rect x="236" y="92" width="36" height="36" rx="0" class="d-box"/>
<rect x="272" y="92" width="36" height="36" rx="0" class="d-b"/>
<text x="290.0" y="115" class="d-muted" text-anchor="middle">k=2</text>
<rect x="308" y="92" width="36" height="36" rx="0" class="d-box"/>
<rect x="200" y="128" width="36" height="36" rx="0" class="d-box"/>
<rect x="236" y="128" width="36" height="36" rx="0" class="d-box"/>
<rect x="272" y="128" width="36" height="36" rx="0" class="d-b"/>
<text x="290.0" y="151" class="d-muted" text-anchor="middle">k=3</text>
<rect x="308" y="128" width="36" height="36" rx="0" class="d-box"/>
<text x="272" y="14" class="d-text d-bold" text-anchor="middle">B (K × N)</text>
<rect x="20" y="186" width="36" height="36" rx="0" class="d-box"/>
<rect x="56" y="186" width="36" height="36" rx="0" class="d-box"/>
<rect x="92" y="186" width="36" height="36" rx="0" class="d-box"/>
<rect x="128" y="186" width="36" height="36" rx="0" class="d-box"/>
<rect x="20" y="222" width="36" height="36" rx="0" class="d-a"/>
<text x="38.0" y="245" class="d-muted" text-anchor="middle">k=0</text>
<rect x="56" y="222" width="36" height="36" rx="0" class="d-a"/>
<text x="74.0" y="245" class="d-muted" text-anchor="middle">k=1</text>
<rect x="92" y="222" width="36" height="36" rx="0" class="d-a"/>
<text x="110.0" y="245" class="d-muted" text-anchor="middle">k=2</text>
<rect x="128" y="222" width="36" height="36" rx="0" class="d-a"/>
<text x="146.0" y="245" class="d-muted" text-anchor="middle">k=3</text>
<rect x="20" y="258" width="36" height="36" rx="0" class="d-box"/>
<rect x="56" y="258" width="36" height="36" rx="0" class="d-box"/>
<rect x="92" y="258" width="36" height="36" rx="0" class="d-box"/>
<rect x="128" y="258" width="36" height="36" rx="0" class="d-box"/>
<rect x="20" y="294" width="36" height="36" rx="0" class="d-box"/>
<rect x="56" y="294" width="36" height="36" rx="0" class="d-box"/>
<rect x="92" y="294" width="36" height="36" rx="0" class="d-box"/>
<rect x="128" y="294" width="36" height="36" rx="0" class="d-box"/>
<text x="92" y="344" class="d-text d-bold" text-anchor="middle">A (M × K)</text>
<rect x="200" y="186" width="36" height="36" rx="0" class="d-box"/>
<rect x="236" y="186" width="36" height="36" rx="0" class="d-box"/>
<rect x="272" y="186" width="36" height="36" rx="0" class="d-box"/>
<rect x="308" y="186" width="36" height="36" rx="0" class="d-box"/>
<rect x="200" y="222" width="36" height="36" rx="0" class="d-box"/>
<rect x="236" y="222" width="36" height="36" rx="0" class="d-box"/>
<rect x="272" y="222" width="36" height="36" rx="0" class="d-c"/>
<rect x="308" y="222" width="36" height="36" rx="0" class="d-box"/>
<rect x="200" y="258" width="36" height="36" rx="0" class="d-box"/>
<rect x="236" y="258" width="36" height="36" rx="0" class="d-box"/>
<rect x="272" y="258" width="36" height="36" rx="0" class="d-box"/>
<rect x="308" y="258" width="36" height="36" rx="0" class="d-box"/>
<rect x="200" y="294" width="36" height="36" rx="0" class="d-box"/>
<rect x="236" y="294" width="36" height="36" rx="0" class="d-box"/>
<rect x="272" y="294" width="36" height="36" rx="0" class="d-box"/>
<rect x="308" y="294" width="36" height="36" rx="0" class="d-box"/>
<text x="272" y="344" class="d-text d-bold" text-anchor="middle">C (M × N)</text>
<text x="290.0" y="245.0" class="d-text" text-anchor="middle">C tile</text>
<path d="M164,240 C182,240 186,240 268,240" class="d-line d-dash" marker-end="url(#d-arrow)"/>
<path d="M290,166 C290,180 290,186 290,222" class="d-line d-dash" marker-end="url(#d-arrow)"/>
<text x="380" y="40" class="d-text" text-anchor="start">For each k = 0, 1, 2, 3:</text>
<text x="380" y="59" class="d-muted" text-anchor="start">1. the block loads A tile (1, k) and</text>
<text x="380" y="78" class="d-muted" text-anchor="start">    B tile (k, 2) into shared memory</text>
<text x="380" y="97" class="d-muted" text-anchor="start">2. __syncthreads()</text>
<text x="380" y="116" class="d-muted" text-anchor="start">3. each thread adds its share of</text>
<text x="380" y="135" class="d-muted" text-anchor="start">    A tile × B tile to registers</text>
<text x="380" y="154" class="d-muted" text-anchor="start">4. __syncthreads(), next k</text>
<text x="380" y="173" class="d-muted" text-anchor="start"></text>
<text x="380" y="192" class="d-text" text-anchor="start">Every input element is read from HBM</text>
<text x="380" y="211" class="d-muted" text-anchor="start">once per tile it meets (N / T times)</text>
<text x="380" y="230" class="d-muted" text-anchor="start">instead of once per output (N times).</text>
</svg>
<figcaption>One thread block computes one tile of C. It walks along a row of A tiles and down a column of B tiles, staging each pair in shared memory, so each loaded value is reused T times from fast memory.</figcaption></figure>

## In an interview

The central kernel. Expect the optimization ladder, tile sizes and the roofline numbers.

Follow-ups to answer out loud once your kernel passes:

- What are the arithmetic intensity and the roofline time of your size? Is it compute-bound?
- Walk through: coalescing, shared-memory tiles, register tiles, vectorized loads, double buffering, Tensor Cores.
- How do you pick BM, BN and BK? What limits them?
