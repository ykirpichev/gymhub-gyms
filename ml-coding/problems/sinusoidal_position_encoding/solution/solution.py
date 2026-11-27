import numpy as np


def solve(seq_len, d_model):
    if seq_len < 0 or d_model < 1:
        raise ValueError("need seq_len >= 0 and d_model >= 1")
    pos = np.arange(seq_len)[:, None]
    even = np.arange(0, d_model, 2)  # 2i for every pair of channels
    angles = pos / 10000.0 ** (even / d_model)  # (seq_len, ceil(d_model / 2))
    pe = np.empty((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[:, : d_model // 2])  # odd d_model: one fewer cos channel
    return pe
