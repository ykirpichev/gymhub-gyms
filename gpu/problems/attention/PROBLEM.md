# Attention

**Level:** hard   **Teaches:** combining matmul and softmax, online softmax, tiling without storing the score matrix

## Task

Compute single-head attention: `O = softmax(Q K^T / sqrt(d)) V`. Each of the `N` query rows of `Q` is scored against all `M` key rows of `K`. The scores become weights through a row softmax, and the output row is the weighted average of the rows of `V`. There is no mask.

## Arguments

| Name | Type           | Shape   | Role   |
| ---- | -------------- | ------- | ------ |
| `Q`  | `const float*` | `N × d` | input  |
| `K`  | `const float*` | `M × d` | input  |
| `V`  | `const float*` | `M × d` | input  |
| `O`  | `float*`       | `N × d` | output |
| `N`  | `int`          |         | size   |
| `M`  | `int`          |         | size   |
| `d`  | `int`          |         | size   |

All are row-major. `Q[i, k]` is at `Q[i * d + k]`, and the same for `K`, `V`, and `O`. `d` is at most 128.

Results must match float32 math within `1e-4`. TF32 tensor-core matmuls are about that far off, so they fail. In Triton, pass `input_precision="ieee"` to `tl.dot`.

## Formula

```
S[i, j] = (sum over k of Q[i, k] * K[j, k]) / sqrt(d)       for j in [0, M)
P[i, j] = exp(S[i, j] - max_j S[i, j]) / sum over j of exp(S[i, j] - max_j S[i, j])
O[i, k] = sum over j of P[i, j] * V[j, k]
```

## Example

```
Q = [[1, 0],     K = [[1, 0],     V = [[1, 2],     d = 2
     [0, 2]]          [0, 1]]          [3, 4]]

S = [[0.7071, 0     ],        P = [[0.6698, 0.3302],
     [0,      1.4142]]             [0.1956, 0.8044]]

O = [[1.6605, 2.6605],
     [2.6089, 3.6089]]
```

Row 0 of `O` is `0.6698 * [1, 2] + 0.3302 * [3, 4]`.

## How to think about it

The plain version is three steps: a matmul for the `N × M` score matrix `S`, a row softmax on `S`, and a second matmul `P V`. You have done each of these before. It is correct, but `S` can be much larger than the inputs: with `N = M = 4096` it is 64 MiB, written once and read twice.

The fast version never stores `S`. Give each block a tile of query rows and walk over `K` and `V` in tiles of key rows. For each key tile, compute the scores, update a running max and a running sum for each query row, and add the weighted `V` rows into an output accumulator. When the running max grows, rescale what you have so far by `exp(old_max - new_max)`. At the end, divide by the sum. This is the online softmax from row softmax, applied to a matmul. It is how FlashAttention works.

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

## In an interview

The flash-attention question: the score matrix, the online softmax, and the memory traffic saved.

Follow-ups to answer out loud once your kernel passes:

- What does the naive version write to HBM, and how large is it at 8K tokens?
- Explain the running max, sum and output rescale.
- Which loop is parallel across blocks and which is sequential within one?
