#### The idea
Speculative decoding is rejection sampling. Accepting the draft token `x` with probability `min(1, p(x)/q(x))` and, on
rejection, sampling from the normalised residual `max(p − q, 0)` produces a token distributed **exactly** as `p`: the
accepted part covers `min(p, q)`, and the residual covers what is left, `p − min(p, q)`.

<figure class="diagram"><svg viewBox="0 0 560 170" width="560" role="img" aria-label="Target and draft distributions, overlap accepted, residual resampled">
<line x1="20" y1="140" x2="540" y2="140" class="d-line"/>
<rect x="60" y="80" width="60" height="60" class="d-a"/><rect x="60" y="40" width="60" height="40" class="d-b"/>
<rect x="200" y="60" width="60" height="80" class="d-a"/>
<rect x="340" y="110" width="60" height="30" class="d-a"/>
<line x1="190" y1="30" x2="270" y2="30" class="d-line" stroke-dasharray="4 3"/>
<line x1="330" y1="50" x2="410" y2="50" class="d-line" stroke-dasharray="4 3"/>
<text x="90" y="158" class="d-text" text-anchor="middle">token a</text>
<text x="230" y="158" class="d-text" text-anchor="middle">token b</text>
<text x="370" y="158" class="d-text" text-anchor="middle">token c</text>
<text x="430" y="70" class="d-muted" text-anchor="start">filled blue: min(p, q), kept</text>
<text x="430" y="90" class="d-muted" text-anchor="start">green: p − q, the residual</text>
<text x="430" y="110" class="d-muted" text-anchor="start">dashes: q above p, rejected</text>
</svg>
<figcaption>Where the draft over-proposes a token (q above p) some proposals are rejected; that mass is handed back to tokens where p exceeds q through the residual.</figcaption></figure>

Batched, the steps are: gather `p_i(x_i)` and `q_i(x_i)`, test all positions at once, count the leading accepts with a
`cumprod`, then pick one distribution row per sequence and sample it.

#### A solution
```python
import torch


def solve(draft_tokens, draft_probs, target_probs, u_accept, u_sample):
    B, K = draft_tokens.shape
    idx = draft_tokens[..., None]
    p_x = target_probs[:, :K].gather(-1, idx).squeeze(-1)             # [B, K]
    q_x = draft_probs.gather(-1, idx).squeeze(-1)
    accept = u_accept * q_x < p_x
    num_accepted = accept.long().cumprod(dim=1).sum(dim=1)            # leading accepts

    rows = torch.arange(B, device=draft_tokens.device)
    j = num_accepted.clamp(max=K - 1)
    residual = (target_probs[rows, j] - draft_probs[rows, j]).clamp(min=0)
    r = torch.where((num_accepted == K)[:, None], target_probs[:, K], residual)   # [B, V]

    cdf = r.cumsum(dim=-1)
    hit = cdf > u_sample[:, None] * cdf[:, -1:]
    next_token = hit.long().argmax(dim=-1)                            # first True
    last_positive = r.shape[-1] - 1 - (r > 0).flip(-1).long().argmax(dim=-1)
    next_token = torch.where(hit.any(dim=-1), next_token, last_positive)
    return num_accepted, next_token
```

The fallback only matters when rounding leaves no `True` at the top of the CDF.

#### What it costs
`O(B·K·V)` for the residual and the cumulative sums, a few hundred MB of reads for the benchmark, and no Python loop.
The point of speculative decoding is upstream: the target model scores `K + 1` positions in one forward pass that
costs about the same as one decode step (decode is memory-bound), so with acceptance rate `α` each pass yields
`(1 − α^(K+1)) / (1 − α)` tokens on average, 2 to 3 times the speed when the draft agrees often. Variants change where
the draft comes from (Medusa heads, EAGLE, n-gram lookup) but keep this verification step.

#### Pitfalls
- Accepting when `p(x) ≥ q(x)` only (greedy agreement): fast, but the output is no longer the target's distribution.
- Sampling the next token from `p_j` instead of the residual after a rejection, which double-counts `min(p, q)`.
- Counting all accepted positions instead of the accepted **prefix**: a token after a rejection must be discarded even
  if its own test passes.
- Forgetting the bonus token when all `K` are accepted.
