#### The idea
Split each byte into its low nibble (`byte & 0xF`, value `2i`) and high nibble (`byte >> 4`, value `2i + 1`), then
sign-extend 4 bits: values 8 to 15 stand for −8 to −1, so subtract 16.

#### A solution
```python
def solve(packed):
    out = []
    for byte in packed:
        for nibble in (byte & 0xF, (byte >> 4) & 0xF):
            out.append(nibble - 16 if nibble >= 8 else nibble)
    return out
```

#### On a real GPU
Kernels never unpack to memory. They load packed bytes (often 32 bits, eight weights, at once), unpack in registers
with shifts and masks, convert to FP16, multiply by the group's scale and feed the matmul. A branch-free sign
extension does it in two operations: `(nibble ^ 8) - 8`, or shift the nibble to the top of a 32-bit int and
arithmetic-shift it back down. Some formats store weights with an offset instead (`0..15` meaning `−8..7`), so the
first question about any int4 checkpoint is which convention it uses.

#### Pitfalls
- Swapping the nibbles: the low nibble comes first.
- Treating values as unsigned 0..15.
