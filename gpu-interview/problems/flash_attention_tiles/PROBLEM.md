# Flash attention, tile by tile

Write `solve(Q, K, V, block, causal)` that computes single-head attention the way flash attention does, without ever building the full score matrix.

- `Q`: `n × d`, `K`: `m × d`, `V`: `m × dv`, as lists of lists or arrays.
- `block`: process keys and values in consecutive blocks of `block` rows (the last may be shorter).
- `causal`: if true, query `i` may only attend to keys `j ≤ i` (then `n == m`).

For each query row keep a running max `m_i`, a running sum `l_i` and an unnormalized output `acc_i`. For each key block:

1. Scores `s = q_i · k_j / sqrt(d)` for the keys in the block (masked keys excluded).
2. `m_new = max(m_i, max(s))`; scale the old state by `alpha = exp(m_i − m_new)`: `l_i ← alpha · l_i + Σ exp(s_j − m_new)` and `acc_i ← alpha · acc_i + Σ exp(s_j − m_new) · v_j`.
3. Skip a block in which every key is masked for this row.

<figure class="diagram"><svg viewBox="0 0 680 240" width="680" role="img" aria-label="Flash attention: one query block streams over key blocks with an online softmax">
<text x="210.0" y="16" class="d-text" text-anchor="middle">Kᵀ blocks (keys and values)</text>
<text x="106" y="134.0" class="d-text" text-anchor="end">Q blocks</text>
<rect x="120" y="22" width="30" height="14" rx="0" class="d-b"/>
<rect x="150" y="22" width="30" height="14" rx="0" class="d-b"/>
<rect x="180" y="22" width="30" height="14" rx="0" class="d-b"/>
<rect x="210" y="22" width="30" height="14" rx="0" class="d-fill"/>
<rect x="240" y="22" width="30" height="14" rx="0" class="d-fill"/>
<rect x="270" y="22" width="30" height="14" rx="0" class="d-fill"/>
<rect x="110" y="44" width="6" height="30" rx="0" class="d-fill"/>
<rect x="110" y="74" width="6" height="30" rx="0" class="d-fill"/>
<rect x="110" y="104" width="6" height="30" rx="0" class="d-a"/>
<rect x="110" y="134" width="6" height="30" rx="0" class="d-fill"/>
<rect x="110" y="164" width="6" height="30" rx="0" class="d-fill"/>
<rect x="110" y="194" width="6" height="30" rx="0" class="d-fill"/>
<rect x="120" y="44" width="30" height="30" rx="0" class="d-box"/>
<path d="M120,44 L150,74" class="d-line"/>
<rect x="150" y="44" width="30" height="30" rx="0" class="d-fill"/>
<rect x="180" y="44" width="30" height="30" rx="0" class="d-fill"/>
<rect x="210" y="44" width="30" height="30" rx="0" class="d-fill"/>
<rect x="240" y="44" width="30" height="30" rx="0" class="d-fill"/>
<rect x="270" y="44" width="30" height="30" rx="0" class="d-fill"/>
<rect x="120" y="74" width="30" height="30" rx="0" class="d-box"/>
<rect x="150" y="74" width="30" height="30" rx="0" class="d-box"/>
<path d="M150,74 L180,104" class="d-line"/>
<rect x="180" y="74" width="30" height="30" rx="0" class="d-fill"/>
<rect x="210" y="74" width="30" height="30" rx="0" class="d-fill"/>
<rect x="240" y="74" width="30" height="30" rx="0" class="d-fill"/>
<rect x="270" y="74" width="30" height="30" rx="0" class="d-fill"/>
<rect x="120" y="104" width="30" height="30" rx="0" class="d-a"/>
<text x="135.0" y="123" class="d-text" text-anchor="middle">1</text>
<rect x="150" y="104" width="30" height="30" rx="0" class="d-a"/>
<text x="165.0" y="123" class="d-text" text-anchor="middle">2</text>
<rect x="180" y="104" width="30" height="30" rx="0" class="d-a"/>
<path d="M180,104 L210,134" class="d-line"/>
<text x="195.0" y="123" class="d-text" text-anchor="middle">3</text>
<rect x="210" y="104" width="30" height="30" rx="0" class="d-fill"/>
<text x="225.0" y="123" class="d-muted" text-anchor="middle">skip</text>
<rect x="240" y="104" width="30" height="30" rx="0" class="d-fill"/>
<text x="255.0" y="123" class="d-muted" text-anchor="middle">skip</text>
<rect x="270" y="104" width="30" height="30" rx="0" class="d-fill"/>
<text x="285.0" y="123" class="d-muted" text-anchor="middle">skip</text>
<rect x="120" y="134" width="30" height="30" rx="0" class="d-box"/>
<rect x="150" y="134" width="30" height="30" rx="0" class="d-box"/>
<rect x="180" y="134" width="30" height="30" rx="0" class="d-box"/>
<rect x="210" y="134" width="30" height="30" rx="0" class="d-box"/>
<path d="M210,134 L240,164" class="d-line"/>
<rect x="240" y="134" width="30" height="30" rx="0" class="d-fill"/>
<rect x="270" y="134" width="30" height="30" rx="0" class="d-fill"/>
<rect x="120" y="164" width="30" height="30" rx="0" class="d-box"/>
<rect x="150" y="164" width="30" height="30" rx="0" class="d-box"/>
<rect x="180" y="164" width="30" height="30" rx="0" class="d-box"/>
<rect x="210" y="164" width="30" height="30" rx="0" class="d-box"/>
<rect x="240" y="164" width="30" height="30" rx="0" class="d-box"/>
<path d="M240,164 L270,194" class="d-line"/>
<rect x="270" y="164" width="30" height="30" rx="0" class="d-fill"/>
<rect x="120" y="194" width="30" height="30" rx="0" class="d-box"/>
<rect x="150" y="194" width="30" height="30" rx="0" class="d-box"/>
<rect x="180" y="194" width="30" height="30" rx="0" class="d-box"/>
<rect x="210" y="194" width="30" height="30" rx="0" class="d-box"/>
<rect x="240" y="194" width="30" height="30" rx="0" class="d-box"/>
<rect x="270" y="194" width="30" height="30" rx="0" class="d-box"/>
<path d="M270,194 L300,224" class="d-line"/>
<path d="M128,146 L202,146" class="d-line" marker-end="url(#d-arrow)"/>
<text x="330" y="30" class="d-text" text-anchor="start">For query block i (here i = 2):</text>
<text x="330" y="49" class="d-muted" text-anchor="start">  m, l, acc ← −∞, 0, 0 in registers / SRAM</text>
<text x="330" y="68" class="d-muted" text-anchor="start">  for key block j = 0 … i (in order 1, 2, 3):</text>
<text x="330" y="87" class="d-muted" text-anchor="start">    s = q kᵀ / √d          (on chip only)</text>
<text x="330" y="106" class="d-muted" text-anchor="start">    m′ = max(m, rowmax(s)); α = exp(m − m′)</text>
<text x="330" y="125" class="d-muted" text-anchor="start">    l = α·l + Σ exp(s − m′)</text>
<text x="330" y="144" class="d-muted" text-anchor="start">    acc = α·acc + exp(s − m′) · v</text>
<text x="330" y="163" class="d-muted" text-anchor="start">  out = acc / l</text>
<text x="330" y="182" class="d-muted" text-anchor="start"></text>
<text x="330" y="201" class="d-text" text-anchor="start">Blocks above the diagonal are skipped (causal);</text>
<text x="330" y="220" class="d-text" text-anchor="start">the diagonal block is masked inside.</text>
<text x="330" y="239" class="d-text" text-anchor="start">The n × m score matrix never reaches HBM.</text>
</svg>
<figcaption>Causal flash attention over a 6 × 6 grid of score tiles. One query block (highlighted) visits key blocks left to right, keeping only a running max, a running sum and an output accumulator per row.</figcaption></figure>

At the end, `out_i = acc_i / l_i` and `lse_i = m_i + log(l_i)`, the log-sum-exp of the row's scores, which the backward pass needs.

Return a tuple `(out, lse)`: an `n × dv` array and a length-`n` array (lists or NumPy arrays both work). Raise `ValueError` if the shapes do not fit, `block < 1`, or `causal` and `n != m`.

```
solve([[1, 0]], [[1, 0], [0, 1]], [[1, 2], [3, 4]], 1, False)
-> ([[1.6604, 2.6604]], [1.1079])
```

Scores are `1/√2` and `0`, so the weights are about `0.670` and `0.330`.

#### What interviewers look for

- The rescale of *both* `l` and `acc` when the max changes; forgetting `acc` is the classic bug.
- Explaining why this saves memory traffic: the `n × m` score matrix never leaves on-chip memory, so HBM traffic is linear in sequence length instead of quadratic.
- Getting the causal mask right at block boundaries, and skipping blocks that are entirely in the future.

#### Follow-ups

- Why does flash attention keep `lse` instead of the full softmax for the backward pass?
- In a GPU kernel, which loop is parallel across thread blocks and which is sequential inside one? (FlashAttention-2 changed this.)
- How would grouped-query attention change which tiles you load?
