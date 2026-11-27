#### The idea
Build an angle matrix with one broadcast: positions as a column `(L, 1)` divided by one frequency per channel pair
`(1, ⌈d/2⌉)`. Sines go into the even channels and cosines into the odd ones. When `d` is odd there is one more sine
column than cosine column, so slice the angles for the cosines.

Each pair of channels is a point rotating on a circle at its own frequency, from one turn every `2π` positions down to
one turn every `2π·10000` positions. Because `PE(pos + k)` is a fixed rotation of `PE(pos)` in each pair, relative
offsets are linear functions of the encoding, which is the property the original paper wanted.

#### A solution
```python
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
```

#### What it costs
`O(L·d)`, computed once and cached. Modern LLMs mostly use RoPE instead: the same rotations applied to queries and keys
inside attention, so the score depends on the relative position directly.

#### Pitfalls
- Using `i` instead of `2i` in the exponent, which changes every frequency.
- Interleaving wrong: putting all sines in the first half and cosines in the second (a valid variant used by some
  codebases, but not this formula).
- Crashing on odd `d_model` with a shape mismatch.
- Integer division producing an integer exponent array.
