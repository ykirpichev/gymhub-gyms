#### The idea
PCA finds the orthogonal directions of largest variance. After centering, the covariance is `XcᵀXc / (N − 1)`; its
eigenvectors are the components and its eigenvalues the variances along them. Equivalently, the right singular vectors
of `Xc` are the components, and the squared singular values are proportional to the variances.

Centering is the step people forget. Without it the first "component" points from the origin to the data's mean.

<figure class="diagram"><svg viewBox="0 0 520 200" width="520" role="img" aria-label="Centered cloud with first and second principal components">
<ellipse cx="260" cy="100" rx="190" ry="45" transform="rotate(-15 260 100)" class="d-fill"/>
<line x1="260" y1="100" x2="440" y2="52" class="d-line" stroke-width="2.5" marker-end="url(#d-arrow)"/>
<line x1="260" y1="100" x2="278" y2="160" class="d-line" stroke-width="2" marker-end="url(#d-arrow)"/>
<circle cx="260" cy="100" r="4" class="d-c"/>
<text x="448" y="48" class="d-text" text-anchor="start">PC1</text>
<text x="286" y="176" class="d-text" text-anchor="start">PC2</text>
<text x="200" y="125" class="d-muted" text-anchor="middle">mean</text>
</svg>
<figcaption>After subtracting the mean, PC1 runs along the long axis of the cloud and PC2 is orthogonal to it. Each eigenvalue is the variance along its axis.</figcaption></figure>

#### A solution
```python
import numpy as np


def solve(X, k):
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be N x D")
    n, d = data.shape
    if not _is_int(k) or k < 1 or k > d or k > n - 1:
        raise ValueError("k must be an integer in 1 .. min(D, N-1)")
    centered = data - data.mean(axis=0)
    if np.all(centered == 0):
        raise ValueError("every row is equal; the components are undefined")
    # Eigenvectors of the Gram matrix, largest eigenvalue first. Dividing by N or N-1
    # does not change the vectors or the explained-variance ratios.
    values, vectors = np.linalg.eigh(centered.T @ centered)
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 0.0, None)
    components = _orient(vectors[:, order][:, :k].T)
    return components, values[:k] / values.sum()


def _orient(components):
    oriented = np.array(components, dtype=np.float64, copy=True)
    for row in oriented:
        index = int(np.argmax(np.abs(row)))
        if row[index] < 0:
            row *= -1
    return oriented


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)
```

`eigh` is for symmetric matrices and returns eigenvalues in ascending order, hence the reversal. Clipping removes tiny
negative eigenvalues that rounding produces for rank-deficient data.

#### What it costs
The `D × D` Gram matrix costs `O(N·D²)` and its eigendecomposition `O(D³)`. A thin SVD of `Xc` costs `O(N·D·min(N, D))`
and is more accurate, since forming `XcᵀXc` squares the condition number. For huge data, randomized SVD gets the top
`k` components in about `O(N·D·k)`.

#### Pitfalls
- Not centering, or centering rows instead of columns.
- Forgetting that `eigh` sorts ascending, and returning the smallest-variance directions.
- Dividing by the sum of the top `k` eigenvalues, so the ratios always sum to 1.
- Returning eigenvectors as columns (`D × k`) instead of rows (`k × D`), or skipping the sign rule.
