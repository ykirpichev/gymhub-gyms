#### The idea
Count per block, then multiply by the number of blocks. There are `(M / BM) × (N / BN)` blocks; each reads a
`BM × K` strip of `A` and a `K × BN` strip of `B`. Intensity is the FLOPs, `2MNK`, over the reads plus one write of `C`.

#### A solution
```python
def solve(M, N, K, BM, BN, elem_bytes):
    if M % BM or N % BN:
        raise ValueError("tiles must divide the matrix")
    blocks = (M // BM) * (N // BN)
    bytes_read = blocks * (BM * K + K * BN) * elem_bytes
    return (bytes_read, 2 * M * N * K / (bytes_read + M * N * elem_bytes))
```

#### The formula behind it
Simplify: `bytes_read = elem_bytes · MNK · (1/BN + 1/BM)`. Each element of `A` is read `N / BN` times and each
element of `B` `M / BM` times. Bigger tiles mean fewer re-reads: the intensity is `2·BM·BN / (BM + BN)` FLOPs per
element read, which for square `B × B` tiles is `B`. With `B = 128` and BF16 that is 64 FLOPs per byte (63 once the
write of `C` is counted, as in the example), still under an H100's ridge point of about 300; caches (L2 holds many tiles)
and larger tiles close the gap. Tile size is capped by shared memory and registers, which is the trade-off
interviewers want to hear.

#### Pitfalls
- Counting `A` and `B` once (`MK + KN`): that is the ideal with infinite on-chip memory, not what tiles achieve.
- Leaving out the write of `C` from the intensity.
