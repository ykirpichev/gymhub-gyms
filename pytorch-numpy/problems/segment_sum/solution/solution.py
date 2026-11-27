import numpy as np


def solve(values, segment_ids, num_segments):
    x = np.asarray(values, dtype=np.float64)
    ids = np.asarray(segment_ids, dtype=np.int64)
    sums = np.zeros((num_segments, x.shape[1]))
    np.add.at(sums, ids, x)
    counts = np.bincount(ids, minlength=num_segments).astype(np.float64)
    means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
    return (sums, means)
