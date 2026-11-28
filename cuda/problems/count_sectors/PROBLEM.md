# Count the sectors

Model the memory system's view of one warp's load. `addresses` is the list of byte addresses the active lanes read
(one per lane, up to 32), and each lane reads `size` bytes starting there.

Return `[sectors, efficiency]`: the number of distinct 32-byte sectors (sector `k` covers bytes `32k` to `32k + 31`)
the warp touches, and the bytes the lanes asked for divided by the bytes moved (`32 × sectors`), rounded to 4 decimals.
Count each requested byte once, even if several lanes read the same address.

Example: 32 lanes reading consecutive floats from address 0, `addresses = [0, 4, 8, …, 124]`, `size = 4`: 4 sectors,
128 / 128 = `[4, 1.0]`.
