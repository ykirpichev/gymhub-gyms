# Count the sectors a warp touches

Write `solve(addresses, size)` for one warp-wide load.

- `addresses`: a list of up to 32 byte addresses, one per lane, or `None` for a lane that is inactive (masked off by a bounds check).
- `size`: bytes each active lane reads (1, 2, 4, 8 or 16).

Each active lane reads bytes `address` to `address + size - 1`. Memory is fetched in 32-byte sectors: sector `s` holds bytes `32s` to `32s + 31`. Return a tuple `(sectors, efficiency)`:

- `sectors`: how many distinct sectors the warp needs.
- `efficiency`: distinct bytes the lanes asked for, divided by `32 * sectors`, a float. Two lanes reading the same byte count it once.

If no lane is active, return `(0, 0.0)`.

```
solve([4 * t for t in range(32)], 4)     ->  (4, 1.0)      # consecutive floats
solve([128 * t for t in range(32)], 4)   ->  (32, 0.125)   # stride of 32 floats
```

(The tests write the addresses out as literal lists.)

#### What interviewers look for

- That you think in sectors, not in individual loads, and handle a read that straddles two sectors.
- Masked lanes cost nothing; lanes reading the same address share a sector.
- Turning the number into a statement: "this kernel needs 8 times the bandwidth it should".

#### Follow-ups

- Where would the `float4` version of the first example land, and why is it still worth doing?
- A kernel reads `x[i + 1]`. Is the misalignment worth fixing?
- How do you see this number in Nsight Compute?
