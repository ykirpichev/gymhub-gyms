#### The idea
Simulate lockstep execution: each step builds the new lane values only from the old ones, so no lane sees a value
another lane wrote in the same step. The offset halves each step: 16, 8, 4, 2, 1.

#### A solution
```python
def solve(values, steps):
    lanes = list(values)
    offset = 16
    for _ in range(steps):
        lanes = [lanes[i] + lanes[i ^ offset] for i in range(32)]   # all lanes at once
        offset //= 2
    return lanes
```

#### On a real GPU
```cuda
for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, offset);
```
After step `s`, every lane holds the sum of a group of `2^s` lanes; after five steps, all 32. Compared with
`__shfl_down_sync`, which leaves the total only in lane 0, the XOR butterfly gives every lane the result, which is
what you want when all lanes need it next (a softmax max or sum, say). No shared memory and no `__syncthreads()`.

#### Pitfalls
- Updating the list in place: lane 16 then reads lane 0's *new* value. Build a new list each step.
- Using `i + offset` instead of `i ^ offset`: lanes near the end read outside the warp.
