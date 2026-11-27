# Broadcasting and einsum in one page

**Broadcasting** lets elementwise operations combine arrays of different shapes without copying. Line the shapes up **from the right**. For each position, the two sizes must be equal, or one of them must be 1 (or missing, which counts as 1). The result takes the larger size; a size-1 axis is stretched with stride 0.

| `a` | `b` | `a + b` |
|---|---|---|
| `(3, 4)` | `(4,)` | `(3, 4)`: `b` is added to every row |
| `(3, 1)` | `(1, 4)` | `(3, 4)`: an outer sum |
| `(8, 1, 6)` | `(7, 1)` | `(8, 7, 6)` |
| `(3, 4)` | `(3,)` | error: 4 and 3 do not match from the right |

<figure class="diagram"><svg viewBox="0 0 680 182" width="680" role="img" aria-label="Broadcasting a (4, 1) column with a (1, 3) row">
<text x="10" y="16" class="d-text d-bold" text-anchor="start">a: (4, 1)</text>
<text x="200" y="16" class="d-text d-bold" text-anchor="start">b: (1, 3)</text>
<text x="440" y="16" class="d-text d-bold" text-anchor="start">a + b: (4, 3)</text>
<rect x="10" y="28" width="30" height="26" rx="3" class="d-a"/>
<text x="25" y="46" class="d-text" text-anchor="middle">0</text>
<rect x="44" y="28" width="30" height="26" rx="3" class="d-fill"/>
<text x="59" y="46" class="d-muted" text-anchor="middle">0</text>
<rect x="78" y="28" width="30" height="26" rx="3" class="d-fill"/>
<text x="93" y="46" class="d-muted" text-anchor="middle">0</text>
<rect x="10" y="58" width="30" height="26" rx="3" class="d-a"/>
<text x="25" y="76" class="d-text" text-anchor="middle">10</text>
<rect x="44" y="58" width="30" height="26" rx="3" class="d-fill"/>
<text x="59" y="76" class="d-muted" text-anchor="middle">10</text>
<rect x="78" y="58" width="30" height="26" rx="3" class="d-fill"/>
<text x="93" y="76" class="d-muted" text-anchor="middle">10</text>
<rect x="10" y="88" width="30" height="26" rx="3" class="d-a"/>
<text x="25" y="106" class="d-text" text-anchor="middle">20</text>
<rect x="44" y="88" width="30" height="26" rx="3" class="d-fill"/>
<text x="59" y="106" class="d-muted" text-anchor="middle">20</text>
<rect x="78" y="88" width="30" height="26" rx="3" class="d-fill"/>
<text x="93" y="106" class="d-muted" text-anchor="middle">20</text>
<rect x="10" y="118" width="30" height="26" rx="3" class="d-a"/>
<text x="25" y="136" class="d-text" text-anchor="middle">30</text>
<rect x="44" y="118" width="30" height="26" rx="3" class="d-fill"/>
<text x="59" y="136" class="d-muted" text-anchor="middle">30</text>
<rect x="78" y="118" width="30" height="26" rx="3" class="d-fill"/>
<text x="93" y="136" class="d-muted" text-anchor="middle">30</text>
<text x="130" y="90" class="d-text d-bold" text-anchor="middle">+</text>
<rect x="200" y="28" width="30" height="26" rx="3" class="d-b"/>
<text x="215" y="46" class="d-text" text-anchor="middle">0</text>
<rect x="234" y="28" width="30" height="26" rx="3" class="d-b"/>
<text x="249" y="46" class="d-text" text-anchor="middle">1</text>
<rect x="268" y="28" width="30" height="26" rx="3" class="d-b"/>
<text x="283" y="46" class="d-text" text-anchor="middle">2</text>
<rect x="200" y="58" width="30" height="26" rx="3" class="d-fill"/>
<text x="215" y="76" class="d-muted" text-anchor="middle">0</text>
<rect x="234" y="58" width="30" height="26" rx="3" class="d-fill"/>
<text x="249" y="76" class="d-muted" text-anchor="middle">1</text>
<rect x="268" y="58" width="30" height="26" rx="3" class="d-fill"/>
<text x="283" y="76" class="d-muted" text-anchor="middle">2</text>
<rect x="200" y="88" width="30" height="26" rx="3" class="d-fill"/>
<text x="215" y="106" class="d-muted" text-anchor="middle">0</text>
<rect x="234" y="88" width="30" height="26" rx="3" class="d-fill"/>
<text x="249" y="106" class="d-muted" text-anchor="middle">1</text>
<rect x="268" y="88" width="30" height="26" rx="3" class="d-fill"/>
<text x="283" y="106" class="d-muted" text-anchor="middle">2</text>
<rect x="200" y="118" width="30" height="26" rx="3" class="d-fill"/>
<text x="215" y="136" class="d-muted" text-anchor="middle">0</text>
<rect x="234" y="118" width="30" height="26" rx="3" class="d-fill"/>
<text x="249" y="136" class="d-muted" text-anchor="middle">1</text>
<rect x="268" y="118" width="30" height="26" rx="3" class="d-fill"/>
<text x="283" y="136" class="d-muted" text-anchor="middle">2</text>
<text x="330" y="90" class="d-text d-bold" text-anchor="middle">=</text>
<rect x="440" y="28" width="30" height="26" rx="3" class="d-c"/>
<text x="455" y="46" class="d-text" text-anchor="middle">0</text>
<rect x="474" y="28" width="30" height="26" rx="3" class="d-c"/>
<text x="489" y="46" class="d-text" text-anchor="middle">1</text>
<rect x="508" y="28" width="30" height="26" rx="3" class="d-c"/>
<text x="523" y="46" class="d-text" text-anchor="middle">2</text>
<rect x="440" y="58" width="30" height="26" rx="3" class="d-c"/>
<text x="455" y="76" class="d-text" text-anchor="middle">10</text>
<rect x="474" y="58" width="30" height="26" rx="3" class="d-c"/>
<text x="489" y="76" class="d-text" text-anchor="middle">11</text>
<rect x="508" y="58" width="30" height="26" rx="3" class="d-c"/>
<text x="523" y="76" class="d-text" text-anchor="middle">12</text>
<rect x="440" y="88" width="30" height="26" rx="3" class="d-c"/>
<text x="455" y="106" class="d-text" text-anchor="middle">20</text>
<rect x="474" y="88" width="30" height="26" rx="3" class="d-c"/>
<text x="489" y="106" class="d-text" text-anchor="middle">21</text>
<rect x="508" y="88" width="30" height="26" rx="3" class="d-c"/>
<text x="523" y="106" class="d-text" text-anchor="middle">22</text>
<rect x="440" y="118" width="30" height="26" rx="3" class="d-c"/>
<text x="455" y="136" class="d-text" text-anchor="middle">30</text>
<rect x="474" y="118" width="30" height="26" rx="3" class="d-c"/>
<text x="489" y="136" class="d-text" text-anchor="middle">31</text>
<rect x="508" y="118" width="30" height="26" rx="3" class="d-c"/>
<text x="523" y="136" class="d-text" text-anchor="middle">32</text>
<text x="10" y="172" class="d-muted" text-anchor="start">Shaded cells are real data; grey ones are the same values seen again through a stride of 0. Nothing is copied.</text>
</svg>
<figcaption>Line shapes up from the right; a dimension of 1 stretches to match. The result is an outer sum of shape (4, 3).</figcaption></figure>

The last row is the classic bug: to add a per-row value, make it `(3, 1)` with `b[:, None]` or `b.unsqueeze(1)`.

> [!WARNING]
> Broadcasting can also *succeed* when you didn't mean it to. `y_pred` of shape `(N, 1)` minus `y` of shape `(N,)` is `(N, N)`, and the mean of that is a number, so the loss runs and is wrong. Check shapes with an `assert` when a reduction follows.

**einsum** writes a contraction by naming axes. Each input gets a letter per axis; letters that appear in the output are kept, letters that don't are summed over:

```python
np.einsum("ij,jk->ik", A, B)          # matmul
np.einsum("bij,bjk->bik", A, B)       # batched matmul
np.einsum("bhqd,bhkd->bhqk", Q, K)    # attention scores per head
np.einsum("ii->i", A)                 # the diagonal
np.einsum("i,j->ij", u, v)            # outer product
np.einsum("ij->", A)                  # sum of everything
```

Reading an einsum is reading the shapes: everything not on the right of `->` is summed.
