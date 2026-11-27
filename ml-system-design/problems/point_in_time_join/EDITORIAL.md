#### The idea
A point-in-time join is an "as-of" lookup: for each event, the latest feature row of the same entity strictly before
the event's time. Index the features once (group by entity, sort each group by time), and each event becomes one
binary search.

<figure class="diagram"><svg viewBox="0 0 560 150" width="560" role="img" aria-label="Feature updates on a timeline and an event that takes the latest earlier value">
<line x1="20" y1="70" x2="540" y2="70" class="d-line" marker-end="url(#d-arrow)"/>
<circle cx="70" cy="70" r="7" class="d-a"/><text x="70" y="50" class="d-text" text-anchor="middle">t=3 "a"</text>
<circle cx="300" cy="70" r="7" class="d-a"/><text x="300" y="50" class="d-text" text-anchor="middle">t=20 "b"</text>
<circle cx="380" cy="70" r="7" class="d-b"/><text x="380" y="50" class="d-text" text-anchor="middle">t=25 "c"</text>
<line x1="380" y1="80" x2="380" y2="110" class="d-line" stroke-dasharray="4 3"/>
<text x="380" y="126" class="d-text d-bold" text-anchor="middle">event at 25</text>
<text x="20" y="145" class="d-muted" text-anchor="start">The event takes "b": "c" has the same timestamp, so it may already contain the event. Using it would leak.</text>
</svg>
<figcaption>One entity's feature history. Each event looks left from its own time, never right and never at its own instant.</figcaption></figure>

#### A solution
```python
import bisect
from collections import defaultdict


def solve(events, features, max_age):
    rows = defaultdict(list)
    for order, (entity, ts, value) in enumerate(features):
        rows[entity].append((ts, order, value))
    index = {}
    for entity, items in rows.items():
        items.sort()  # by time, then by input order: the last of equal times comes last
        index[entity] = ([ts for ts, _, _ in items], [value for _, _, value in items])
    out = []
    for entity, ts in events:
        if entity not in index:
            out.append(None)
            continue
        times, values = index[entity]
        i = bisect.bisect_left(times, ts)  # rows strictly before ts
        if i == 0 or ts - times[i - 1] > max_age:
            out.append(None)
        else:
            out.append(values[i - 1])
    return out
```

Sorting `(ts, order, value)` triples keeps equal timestamps in input order, the times sit in a separate list for
`bisect`, and `bisect_left` returns exactly the count of rows with `ts < event_ts`.

#### What it costs
`O(F log F)` to sort the features and `O(E log F)` for the lookups. Scanning every feature row per event is
`O(E·F)`, hopeless at the scale of a training set. In a warehouse the same join is a window function (the last value
per entity ordered by time, over rows before the event), or Spark's or pandas' `merge_asof`, which sort both sides and
sweep them together in `O((E + F) log)`. Feature stores (Feast, Tecton, Michelangelo) implement exactly this when they
build training sets.

#### Pitfalls
- `<=` instead of `<`: a value computed in the same second can include the label event itself.
- Joining on the entity alone (latest value overall), which leaks the future into every older event.
- Forgetting staleness: a value from six months ago is often worse than "missing", and the model should learn that it
  is missing.
- An unstable sort, which picks an arbitrary row among equal timestamps.
