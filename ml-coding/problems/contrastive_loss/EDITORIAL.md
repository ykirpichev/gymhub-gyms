#### The idea
In a batch of `N` pairs, each image's matching caption is the right "class" among `N` captions, and vice versa. So the
loss is ordinary cross-entropy with the **diagonal** as the targets, computed once over rows (image → text) and once
over columns (text → image), then averaged.

<figure class="diagram"><svg viewBox="0 0 440 190" width="440" role="img" aria-label="N by N similarity matrix with positives on the diagonal">
<text x="150" y="18" class="d-muted" text-anchor="middle">captions →</text>
<rect x="60" y="28" width="40" height="36" class="d-a"/><rect x="102" y="28" width="40" height="36" class="d-box"/><rect x="144" y="28" width="40" height="36" class="d-box"/><rect x="186" y="28" width="40" height="36" class="d-box"/>
<rect x="60" y="66" width="40" height="36" class="d-box"/><rect x="102" y="66" width="40" height="36" class="d-a"/><rect x="144" y="66" width="40" height="36" class="d-box"/><rect x="186" y="66" width="40" height="36" class="d-box"/>
<rect x="60" y="104" width="40" height="36" class="d-box"/><rect x="102" y="104" width="40" height="36" class="d-box"/><rect x="144" y="104" width="40" height="36" class="d-a"/><rect x="186" y="104" width="40" height="36" class="d-box"/>
<rect x="60" y="142" width="40" height="36" class="d-box"/><rect x="102" y="142" width="40" height="36" class="d-box"/><rect x="144" y="142" width="40" height="36" class="d-box"/><rect x="186" y="142" width="40" height="36" class="d-a"/>
<text x="50" y="110" class="d-muted" text-anchor="end">images</text>
<text x="250" y="70" class="d-text" text-anchor="start">rows: image → text CE</text>
<text x="250" y="96" class="d-text" text-anchor="start">columns: text → image CE</text>
<text x="250" y="122" class="d-muted" text-anchor="start">filled: positives (targets)</text>
<text x="250" y="142" class="d-muted" text-anchor="start">the rest: in-batch negatives</text>
</svg>
<figcaption>CLIP's logits: cosine similarity over temperature. Every off-diagonal entry is a free negative, so bigger batches give a harder task.</figcaption></figure>

Normalise first, so the logits are cosine similarities over the temperature, then use log-sum-exp for both directions.

#### A solution
```python
import numpy as np


def _normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _cross_entropy_diagonal(logits):
    m = logits.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    return float(np.mean(lse - np.diag(logits)))


def solve(img, txt, temperature):
    img = np.asarray(img, dtype=np.float64)
    txt = np.asarray(txt, dtype=np.float64)
    if img.ndim != 2 or img.shape != txt.shape or not temperature > 0:
        raise ValueError("img and txt must both be N x d, and temperature must be positive")
    logits = _normalize(img) @ _normalize(txt).T / temperature
    return (_cross_entropy_diagonal(logits) + _cross_entropy_diagonal(logits.T)) / 2
```

#### What it costs
`O(N²·d)` for the similarity matrix and `O(N²)` memory. CLIP trained with `N = 32768`, so the matrix is sharded across
GPUs: each device computes its rows against all gathered text embeddings. SigLIP replaces the softmax with an
independent sigmoid per pair, which removes the need for the full row normalisation.

#### Pitfalls
- Forgetting to normalise, so the loss depends on embedding length.
- Computing only one direction.
- `np.log(softmax)` with temperature `1e-4`: logits of `1e4` overflow `exp`.
- Treating the temperature as a multiplier instead of a divisor (CLIP learns `log(1/T)` and clips it at 100).
