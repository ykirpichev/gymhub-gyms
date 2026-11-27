#### The idea
The number of rooms needed is the peak number of meetings running at once. Sweep the events in time order with a
counter: `+1` at each start, `−1` at each end. Sorting starts and ends as two separate arrays and walking them with two
pointers does exactly that, and processing ends first on ties lets a room freed at `t` be reused at `t`.

<figure class="diagram"><svg viewBox="0 0 560 170" width="560" role="img" aria-label="Meetings on a timeline with the running count of rooms">
<line x1="20" y1="140" x2="540" y2="140" class="d-line"/>
<rect x="30" y="20" width="480" height="18" rx="3" class="d-a"/><text x="270" y="33" class="d-text" text-anchor="middle">[0, 30)</text>
<rect x="110" y="44" width="80" height="18" rx="3" class="d-b"/><text x="150" y="57" class="d-text" text-anchor="middle">[5, 10)</text>
<rect x="190" y="44" width="80" height="18" rx="3" class="d-b"/><text x="230" y="57" class="d-text" text-anchor="middle">[10, 15)</text>
<rect x="270" y="44" width="80" height="18" rx="3" class="d-b"/><text x="310" y="57" class="d-text" text-anchor="middle">[15, 20)</text>
<text x="30" y="100" class="d-muted" text-anchor="start">running count:</text>
<text x="70" y="120" class="d-text" text-anchor="middle">1</text>
<text x="150" y="120" class="d-text" text-anchor="middle">2</text>
<text x="230" y="120" class="d-text" text-anchor="middle">2</text>
<text x="310" y="120" class="d-text" text-anchor="middle">2</text>
<text x="430" y="120" class="d-text" text-anchor="middle">1</text>
<text x="20" y="162" class="d-muted" text-anchor="start">At 10 and 15 one meeting ends as another starts: end first, so the count never reaches 3.</text>
</svg>
<figcaption>The answer is the peak of the running count over the sweep.</figcaption></figure>

#### A solution
```python
def solve(meetings):
    starts = sorted(s for s, _ in meetings)
    ends = sorted(e for _, e in meetings)
    rooms = best = j = 0
    for s in starts:
        while ends[j] <= s:        # every meeting that ended by s frees its room
            j += 1
            rooms -= 1
        rooms += 1
        best = max(best, rooms)
    return best
```

`ends[j]` never runs past the end: there are as many ends as starts, and each start's own end is greater than it.

#### What it costs
`O(n log n)` for the two sorts, `O(n)` for the sweep. The heap version (sort by start, keep a min-heap of end times in
use) has the same complexity and also tells you **which** room each meeting gets, which is what you need to actually
assign rooms. The sweep-and-count pattern answers "maximum overlap" questions everywhere: concurrent sessions, peak
memory of a set of allocations, or how many GPUs a batch of jobs needs at once.

#### Pitfalls
- Processing starts before ends at equal times: back-to-back meetings get separate rooms.
- Answering with the number of overlapping **pairs**, or with the length of the longest chain.
- Sorting meetings by end and greedily counting non-overlapping ones (that is interval scheduling, a different
  question).
