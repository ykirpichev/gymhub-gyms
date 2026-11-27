# Compute strides after a permute

Write `solve(shape, perm)` that returns the shape and the strides (in elements, as PyTorch's `.stride()` reports them) of `torch.empty(shape).permute(perm)`, and whether the result is contiguous, **without using PyTorch or NumPy**.

- `shape`: a list of positive ints.
- `perm`: a permutation of `0 .. len(shape) − 1`.

The input is contiguous (row-major), so its stride for axis `i` is the product of `shape[i+1:]`. A permute reorders the axes and their strides together. The result is contiguous when its strides equal the row-major strides of its own shape (ignoring axes of size 1, whose stride never matters).

Return `(new_shape, new_strides, contiguous)` with lists for the first two.

```
solve([2, 3, 4], [0, 2, 1])  ->  ([2, 4, 3], [12, 1, 4], False)
```

#### What interviewers look for

- Row-major strides as suffix products, computed right to left.
- Knowing that permute moves no data, so `view` after it fails and `reshape` copies.

#### Follow-ups

- Why do attention implementations call `.transpose(1, 2).contiguous()` before `.view(B, T, C)`?
- What are the strides of `x.expand(4, 3)` for `x` of shape `(1, 3)`? (A zero stride: every row is the same memory.)
- What is "channels last" memory format, and why do convolutions on GPUs prefer it?
