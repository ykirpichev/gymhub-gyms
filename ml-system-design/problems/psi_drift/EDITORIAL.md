#### The idea
PSI is a symmetric relative-entropy score between two histograms: each bin adds `(a − e)·ln(a/e)`, which is never
negative (both factors have the same sign), and is large when a bin's share changes by a large **ratio**. That is why
empty bins need a floor: `ln(a/0)` is infinite.

<figure class="diagram"><svg viewBox="0 0 560 170" width="560" role="img" aria-label="Baseline and current histograms side by side per bin">
<line x1="20" y1="140" x2="540" y2="140" class="d-line"/>
<rect x="50" y="90" width="40" height="50" class="d-a"/><rect x="92" y="90" width="40" height="50" class="d-b"/>
<rect x="170" y="90" width="40" height="50" class="d-a"/><rect x="212" y="139" width="40" height="1" class="d-b"/>
<rect x="290" y="90" width="40" height="50" class="d-a"/><rect x="332" y="90" width="40" height="50" class="d-b"/>
<rect x="410" y="90" width="40" height="50" class="d-a"/><rect x="452" y="40" width="40" height="100" class="d-b"/>
<text x="91" y="158" class="d-text" text-anchor="middle">bin 0</text>
<text x="211" y="158" class="d-text" text-anchor="middle">bin 1</text>
<text x="331" y="158" class="d-text" text-anchor="middle">bin 2</text>
<text x="451" y="158" class="d-text" text-anchor="middle">bin 3</text>
<text x="20" y="22" class="d-muted" text-anchor="start">blue: baseline share e, green: current share a. Bin 1 empties out and dominates PSI.</text>
</svg>
<figcaption>The example: equal baseline bins, a current sample that abandons bin 1 and piles into bin 3.</figcaption></figure>

#### A solution
```python
import bisect
import math


def _shares(values, edges):
    counts = [0] * (len(edges) + 1)
    for x in values:
        counts[bisect.bisect_right(edges, x)] += 1
    return [c / len(values) or 0.0001 for c in counts]


def solve(baseline, current, edges):
    e = _shares(baseline, edges)
    a = _shares(current, edges)
    psi = sum((ai - ei) * math.log(ai / ei) for ai, ei in zip(a, e))
    verdict = "stable" if psi < 0.1 else "moderate" if psi < 0.25 else "major"
    return psi, verdict
```

`c / len(values) or 0.0001` floors exactly-zero shares and nothing else.

#### What it costs
`O((B + C) log m)` for the binning. In production the baseline histogram is computed once at training time and stored
with the model; each monitoring window only bins the current values. For high-dimensional inputs (embeddings, text),
teams monitor a few summary statistics, the prediction distribution, or a domain classifier trained to tell old data
from new (if it can, the data has drifted).

#### Pitfalls
- Bins from the current data instead of fixed baseline bins, so the reference moves every week.
- `bisect_left`, which sends values equal to an edge to the left bin.
- Renormalising after flooring the zeros (harmless in size, but not this spec, and it changes the thresholds).
- Treating a high PSI as proof the model is worse. It is a signal to investigate; check the online metric by segment.
