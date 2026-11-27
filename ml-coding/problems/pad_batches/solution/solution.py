import numpy as np


def solve(sequences, pad_id=0, max_len=None):
    if max_len is not None and (isinstance(max_len, bool) or not isinstance(max_len, int) or max_len < 0):
        raise ValueError("max_len must be None or an int >= 0")
    if len(sequences) == 0:
        return np.zeros((0, 0), dtype=np.int64), np.zeros((0, 0), dtype=bool), []

    rows = []
    for sequence in sequences:
        row = [int(token) for token in sequence]
        if max_len is not None:
            row = row[:max_len]
        rows.append(row)
    width = max(len(row) for row in rows)
    ids = np.full((len(rows), width), int(pad_id), dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=bool)
    lengths = []
    for index, row in enumerate(rows):
        length = len(row)
        lengths.append(length)
        if length:
            ids[index, :length] = row
            mask[index, :length] = True
    return ids, mask, lengths
