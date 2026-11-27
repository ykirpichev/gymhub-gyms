#### The idea
Each active lane touches the sectors from `address // 32` to `(address + size - 1) // 32`; the warp needs the union
of them. Efficiency is the distinct bytes asked for over the bytes fetched, `32 × sectors`.

#### A solution
```python
def solve(addresses, size):
    sectors, wanted = set(), set()
    for address in addresses:
        if address is None:
            continue
        sectors.update(range(address // 32, (address + size - 1) // 32 + 1))
        wanted.update(range(address, address + size))
    if not sectors:
        return (0, 0.0)
    return (len(sectors), len(wanted) / (32 * len(sectors)))
```

#### On a real GPU
This is the number Nsight Compute shows as *sectors per request*. Ideal for 4-byte loads is 4 (128 bytes); a stride
of 32 floats gives 32, and the kernel needs 8 times the bandwidth it seems to. A misaligned contiguous load
(`x[i + 1]`) costs 5 sectors instead of 4, which is why vectorized `float4` loads require 16-byte alignment.

#### Pitfalls
- A lane can straddle two sectors (an 8-byte load at address 28): count both.
- Two lanes reading the same bytes count those bytes once, both for sectors and for efficiency.
