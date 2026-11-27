#### The idea
The mask is the AND of three conditions, and each is a broadcastable array:

- causal: `tril(ones(L, L))`, shape `(L, L)`, true where `j ≤ i`;
- key is real: `valid[:, None, :]`, shape `(B, 1, L)`, a column condition;
- query is real: `valid[:, :, None]`, shape `(B, L, 1)`, a row condition.

`&` broadcasts them to `(B, L, L)` without a loop.

<figure class="diagram"><svg viewBox="0 0 520 170" width="520" role="img" aria-label="Causal mask for a sequence whose last position is padding">
<text x="10" y="20" class="d-text d-bold" text-anchor="start">valid = [T, T, T, F]</text>
<text x="200" y="20" class="d-muted" text-anchor="start">keys j →</text>
<rect x="200" y="30" width="30" height="30" class="d-a"/><rect x="232" y="30" width="30" height="30" class="d-box"/><rect x="264" y="30" width="30" height="30" class="d-box"/><rect x="296" y="30" width="30" height="30" class="d-box"/>
<rect x="200" y="62" width="30" height="30" class="d-a"/><rect x="232" y="62" width="30" height="30" class="d-a"/><rect x="264" y="62" width="30" height="30" class="d-box"/><rect x="296" y="62" width="30" height="30" class="d-box"/>
<rect x="200" y="94" width="30" height="30" class="d-a"/><rect x="232" y="94" width="30" height="30" class="d-a"/><rect x="264" y="94" width="30" height="30" class="d-a"/><rect x="296" y="94" width="30" height="30" class="d-box"/>
<rect x="200" y="126" width="30" height="30" class="d-box"/><rect x="232" y="126" width="30" height="30" class="d-box"/><rect x="264" y="126" width="30" height="30" class="d-box"/><rect x="296" y="126" width="30" height="30" class="d-box"/>
<text x="190" y="50" class="d-muted" text-anchor="end">i = 0</text>
<text x="190" y="146" class="d-muted" text-anchor="end">i = 3 (pad)</text>
<text x="340" y="80" class="d-muted" text-anchor="start">filled: may attend</text>
<text x="340" y="100" class="d-muted" text-anchor="start">padding row and column: all False</text>
</svg>
<figcaption>Lower triangle for causality, then the padding query's row and the padding key's column are cleared.</figcaption></figure>

#### A solution
```python
import numpy as np


def solve(valid):
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2:  # an empty batch [] arrives as shape (0,)
        valid = valid.reshape(len(valid), 0)
    length = valid.shape[1]
    causal = np.tril(np.ones((length, length), dtype=bool))  # causal[i, j] = j <= i
    return causal & valid[:, :, None] & valid[:, None, :]
```

The reshape handles `valid = []`, which NumPy reads as shape `(0,)` rather than `(0, L)`.

#### What it costs
`O(B·L²)` booleans. Real kernels never build this tensor: FlashAttention takes a causal flag and per-sequence lengths
and skips whole masked tiles, which also halves the work for causal attention.

#### Pitfalls
- `np.triu` instead of `np.tril` (that is the "future" mask).
- Masking only padding keys. A padding query with a valid key gets a finite row; harmless for the loss if ignored, but
  wrong against the spec, and a padding row with **no** valid key becomes `nan` after softmax in many implementations.
- Assuming right padding and using lengths: left padding and holes break it.
- Returning an int or float mask instead of bool.
