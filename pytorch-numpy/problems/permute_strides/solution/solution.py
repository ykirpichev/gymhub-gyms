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
    contiguous = all(size == 1 or got == want for size, got, want in zip(new_shape, new_strides, expected))
    return (new_shape, new_strides, contiguous)
