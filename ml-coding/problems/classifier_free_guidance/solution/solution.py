import numpy as np


def solve(cond, uncond, w):
    scale = float(w)
    if uncond is None:
        batch = np.asarray(cond, dtype=np.float64)
        if batch.ndim == 0 or batch.shape[0] % 2 != 0:
            raise ValueError("a stacked batch must have an even length on axis 0")
        half = batch.shape[0] // 2
        conditional = batch[:half]
        unconditional = batch[half:]
    else:
        conditional = np.asarray(cond, dtype=np.float64)
        unconditional = np.asarray(uncond, dtype=np.float64)
        if conditional.shape != unconditional.shape:
            raise ValueError("cond and uncond must have the same shape")
    return unconditional + scale * (conditional - unconditional)
