#### The idea
A permute moves no data. It reorders the shape and the strides *together*, so the new axis `k` is old axis
`perm[k]` with its old stride. Contiguity then means: do these strides equal the row-major strides of the new
shape? Axes of size 1 are skipped, because their stride is never used to step anywhere.

#### A solution
```python
def _row_major(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def solve(shape, perm):
    strides = _row_major(shape)
    new_shape = [shape[p] for p in perm]
    new_strides = [strides[p] for p in perm]
    expected = _row_major(new_shape)
    contiguous = all(n == 1 or got == want for n, got, want in zip(new_shape, new_strides, expected))
    return (new_shape, new_strides, contiguous)
```

#### Why it matters
`x.permute(0, 2, 1)` is free, but the next operation that needs contiguous memory (`.view`, many custom kernels,
`torch.compile` graph breaks aside) either fails or calls `.contiguous()`, which copies. In attention code,
`q.view(B, T, H, D).transpose(1, 2)` is the classic case: the result is `(B, H, T, D)` with strides of the original
layout, so a later `.view(B * H, T, D)` raises and `.reshape` silently copies.

#### Pitfalls
- Permuting the shape and recomputing row-major strides for it: that describes a copy, not the view.
- Calling a single-element or size-1 axis a contiguity breaker: PyTorch ignores their strides too.
