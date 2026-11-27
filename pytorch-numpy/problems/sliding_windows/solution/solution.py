import numpy as np


def solve(x, window, step):
    if window < 1 or step < 1:
        raise ValueError("window and step must be >= 1")
    a = np.asarray(x)
    if window > len(a):
        return (np.empty((0, window), dtype=a.dtype), np.empty(0))
    windows = np.lib.stride_tricks.sliding_window_view(a, window)[::step]
    return (windows, windows.mean(axis=1))
