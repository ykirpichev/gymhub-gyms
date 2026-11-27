import numpy as np


def solve(valid):
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2:  # an empty batch [] arrives as shape (0,)
        valid = valid.reshape(len(valid), 0)
    length = valid.shape[1]
    causal = np.tril(np.ones((length, length), dtype=bool))  # causal[i, j] = j <= i
    return causal & valid[:, :, None] & valid[:, None, :]
