## Hint 1
Truncate each row with `row[:max_len]` before measuring the width. The width is `max(len(row) for row in rows)` after that cut. A `max_len` larger than every sequence does not add extra columns.

## Hint 2
Right-pad by writing tokens at `ids[i, :n]`. Build the mask from those lengths, not from `ids != pad_id`: a real token may equal the padding id. `lengths` counts tokens kept after truncation, not the original lengths.

## Hint 3
An empty batch is `np.zeros((0, 0), dtype=np.int64)` and a boolean array of the same shape, with `lengths == []`. A non-empty batch whose rows are all empty has shape `(N, 0)`. `ids` must be int64 and `mask` must be bool.
