#### The idea
Truncate first, then measure the width, then allocate once: `ids = full((N, W), pad_id)` and a `False` mask, and copy
each row into its left part. The mask, not the token value, says what is real, because a real token can equal
`pad_id`.

#### A solution
```python
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
```

#### What it costs
`O(N·W)` memory, and that is the real cost of padding: every padded slot is computed and thrown away. A batch with one
2,000-token sequence and thirty 50-token ones is 95% padding. Training pipelines bucket sequences by length; inference
servers avoid padding altogether by **packing** sequences into one flat token array with offsets (`cu_seqlens` in
FlashAttention's varlen API) and by continuous batching.

#### Pitfalls
- `mask = ids != pad_id`: wrong whenever a real token equals the pad id.
- Setting the width to `max_len` even when every sequence is shorter.
- Truncating from the left (keeping the end) when the spec says to keep the beginning. Both exist in practice; for
  chat models people often keep the end of the conversation.
- `np.array(sequences)` on ragged lists, which builds an object array or raises.
