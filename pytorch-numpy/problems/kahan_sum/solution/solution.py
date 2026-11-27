import numpy as np


def solve(values):
    naive = np.float32(0.0)
    total = np.float32(0.0)
    c = np.float32(0.0)
    for value in values:
        x = np.float32(value)
        naive = np.float32(naive + x)
        y = np.float32(x - c)
        t = np.float32(total + y)
        c = np.float32(np.float32(t - total) - y)
        total = t
    return (float(naive), float(total))
