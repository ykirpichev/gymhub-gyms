#### The idea
The `attention` kernel per head, with query `i` stopping at key `i`. The benchmark rewards skipping the masked half
entirely rather than computing scores and setting them to −∞: about half the FLOPs disappear.

#### A solution
Reuse the warp-per-row kernel from `attention` with two changes:

```cuda
int h = blockIdx.y;                                  // grid.y = H heads
const float* Qh = Q + (size_t)h * L * d;             // same offset for K, V and O
...
for (int j = 0; j <= row; ++j) { ... }               // keys 0 .. row only
```
Row 0 sees only itself, so `O[0] = V[0]`, a quick check of your indexing.

In a tiled (flash) version, a query tile `[i0, i0 + Bq)` visits key tiles `[j0, j0 + Bk)` with `j0 <= i0 + Bq − 1`:
tiles fully below the diagonal need no mask, the diagonal tile needs a per-element mask, and tiles above are
skipped. Launch the *last* query tiles first: they have the most work, and the GPU fills in with short ones at the
end (FlashAttention-2 does this).

#### What it costs
`2HL²d` FLOPs, half of the unmasked version. Per head, the minimum traffic is `16Ld` bytes; with `L` in the
thousands the kernel is compute-bound when tiled and memory-bound in this simple warp form.

#### Pitfalls
- Off by one at the diagonal: the mask is `j > i`, so each token attends to itself.
- Mixing up the head offset: row `i` of head `h` is at `(h * L + i) * d`.
- Masking with a large negative number and then subtracting the max: fine; masking *before* scaling with `-1e9` in
  FP16 overflows. Skipping masked keys avoids the question.
