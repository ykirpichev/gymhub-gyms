## Hint 1

`strides = [1] * n`, then for `i` from `n − 2` down to 0, `strides[i] = strides[i + 1] * shape[i + 1]`.
