#### The idea
The median splits the sorted numbers into two halves. Keep each half in a heap, oriented so its top is the element next
to the middle:

<figure class="diagram"><svg viewBox="0 0 560 120" width="560" role="img" aria-label="Lower half in a max-heap and upper half in a min-heap, tops meet at the median">
<rect x="30" y="30" width="230" height="40" rx="4" class="d-a"/><text x="145" y="55" class="d-text" text-anchor="middle">lower half: max-heap (top = largest)</text>
<rect x="300" y="30" width="230" height="40" rx="4" class="d-b"/><text x="415" y="55" class="d-text" text-anchor="middle">upper half: min-heap (top = smallest)</text>
<line x1="280" y1="20" x2="280" y2="80" class="d-line" stroke-dasharray="4 3"/>
<text x="280" y="100" class="d-text d-bold" text-anchor="middle">median: lower top, or the mean of both tops</text>
</svg>
<figcaption>Invariant: every lower element ≤ every upper element, and the lower half has the same size or one more.</figcaption></figure>

Each arrival: push into the lower half, pass its largest to the upper half (this keeps the order invariant), and if the
upper half became larger, pass its smallest back (this keeps the size invariant).

#### A solution
```python
import heapq


def solve(nums):
    low, high = [], []            # low stores negated values: a max-heap
    out = []
    for x in nums:
        heapq.heappush(low, -x)
        heapq.heappush(high, -heapq.heappop(low))
        if len(high) > len(low):
            heapq.heappush(low, -heapq.heappop(high))
        if len(low) > len(high):
            out.append(float(-low[0]))
        else:
            out.append((-low[0] + high[0]) / 2.0)
    return out
```

#### What it costs
`O(log n)` per arrival and `O(n)` memory. A sorted list with `bisect.insort` is `O(n)` per insert (fast memmove in
practice, quadratic in theory). For a **sliding-window** median you also need deletions: lazy deletion in both heaps,
or a balanced tree / `SortedList`. For approximate quantiles over huge streams (p50 and p99 latency on a serving
fleet), t-digest and KLL sketches use fixed memory and can be merged across machines.

#### Pitfalls
- Forgetting to negate on the way back out of the max-heap.
- Pushing the new number into whichever heap is smaller without the pass-through step, which breaks the ordering
  invariant.
- Integer division for the even case (`//`), which truncates `-1.5` to `-2`.
- Sorting the list after every arrival.
