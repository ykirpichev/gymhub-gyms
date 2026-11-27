#### The idea
Sort by start. After sorting, any interval that overlaps the current merged block must start before (or at) the block's
end, because every later interval starts even further right. So one left-to-right sweep suffices: extend the last
block when the next interval starts at or before its end, otherwise open a new block.

<figure class="diagram"><svg viewBox="0 0 560 150" width="560" role="img" aria-label="Sorted intervals swept left to right into merged blocks">
<line x1="20" y1="120" x2="540" y2="120" class="d-line"/>
<rect x="40" y="20" width="120" height="16" rx="3" class="d-a"/><text x="100" y="32" class="d-text" text-anchor="middle">[1, 4]</text>
<rect x="160" y="42" width="40" height="16" rx="3" class="d-a"/><text x="180" y="54" class="d-text" text-anchor="middle">[4, 5]</text>
<rect x="80" y="64" width="40" height="16" rx="3" class="d-a"/><text x="100" y="76" class="d-text" text-anchor="middle">[2, 3]</text>
<rect x="320" y="20" width="120" height="16" rx="3" class="d-b"/><text x="380" y="32" class="d-text" text-anchor="middle">[8, 11]</text>
<rect x="40" y="94" width="160" height="16" rx="3" class="d-c"/><text x="120" y="106" class="d-text" text-anchor="middle">merged [1, 5]</text>
<rect x="320" y="94" width="120" height="16" rx="3" class="d-c"/><text x="380" y="106" class="d-text" text-anchor="middle">merged [8, 11]</text>
<text x="20" y="140" class="d-muted" text-anchor="start">Touching ends (4 and 4) merge; the nested [2, 3] changes nothing; a gap starts a new block.</text>
</svg>
<figcaption>After sorting by start, each interval either extends the current block or starts a new one.</figcaption></figure>

#### A solution
```python
def solve(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:       # overlaps or touches the current block
            merged[-1][1] = max(merged[-1][1], end)  # max: a nested interval must not shrink it
        else:
            merged.append([start, end])
    return merged
```

#### What it costs
`O(n log n)` for the sort and `O(n)` for the sweep; `O(n)` extra memory for the output. The same sort-then-sweep shape
solves meeting rooms (count overlaps with a sweep over start and end events), interval insertion, and the "minimum
number of arrows" family.

#### Pitfalls
- `start < end` instead of `<=`: touching intervals stay separate.
- `merged[-1][1] = end` without `max`: a nested interval shrinks the block.
- Mutating the input lists in place (the solution appends fresh `[start, end]` lists).
- Comparing every pair: `O(n²)` and still needs repeated passes to reach a fixed point.
