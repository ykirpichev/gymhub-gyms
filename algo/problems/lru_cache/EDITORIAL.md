#### The idea
You need two things in `O(1)`: find a key, and know which key is least recent. A hash map gives the first. A
**doubly linked list** in recency order gives the second: move a node to the front on every use, and evict from the
back. The map stores pointers to the list nodes so a hit can unlink its node in `O(1)`.

<figure class="diagram"><svg viewBox="0 0 600 150" width="600" role="img" aria-label="Hash map pointing into a doubly linked list ordered by recency">
<text x="20" y="20" class="d-text d-bold" text-anchor="start">hash map: key → node</text>
<rect x="20" y="30" width="60" height="26" rx="3" class="d-box"/><text x="50" y="48" class="d-text" text-anchor="middle">7</text>
<rect x="20" y="58" width="60" height="26" rx="3" class="d-box"/><text x="50" y="76" class="d-text" text-anchor="middle">3</text>
<rect x="20" y="86" width="60" height="26" rx="3" class="d-box"/><text x="50" y="104" class="d-text" text-anchor="middle">9</text>
<rect x="180" y="60" width="80" height="34" rx="4" class="d-a"/><text x="220" y="82" class="d-text" text-anchor="middle">3 (newest)</text>
<rect x="320" y="60" width="80" height="34" rx="4" class="d-a"/><text x="360" y="82" class="d-text" text-anchor="middle">9</text>
<rect x="460" y="60" width="100" height="34" rx="4" class="d-b"/><text x="510" y="82" class="d-text" text-anchor="middle">7 (evict next)</text>
<line x1="262" y1="77" x2="318" y2="77" class="d-line" marker-end="url(#d-arrow)"/>
<line x1="402" y1="77" x2="458" y2="77" class="d-line" marker-end="url(#d-arrow)"/>
<line x1="82" y1="43" x2="458" y2="70" class="d-line" stroke-dasharray="4 3"/>
<line x1="82" y1="71" x2="178" y2="74" class="d-line" stroke-dasharray="4 3"/>
<line x1="82" y1="99" x2="318" y2="84" class="d-line" stroke-dasharray="4 3"/>
<text x="20" y="140" class="d-muted" text-anchor="start">get/put: look up the node, unlink it, relink it at the front. Full: drop the back node and its key.</text>
</svg>
<figcaption>An LRU cache is a hash map into a doubly linked list kept in recency order.</figcaption></figure>

In Python, `collections.OrderedDict` is exactly this structure (a dict plus a linked list), with `move_to_end` and
`popitem(last=False)`.

#### A solution
```python
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = OrderedDict()  # oldest first, newest last

    def get(self, key):
        if key not in self.data:
            return -1
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)
```

#### What it costs
`O(1)` per operation and `O(capacity)` memory. In C++, the standard version is `std::list<pair<int,int>>` for the order
plus `unordered_map<int, list<...>::iterator>`, with `list::splice` to move a node to the front without allocating.
Real caches (Redis, CPU caches) approximate LRU with sampling or clock bits, because exact recency on every access is
too expensive under concurrency.

#### Pitfalls
- A list of keys for the order: `list.remove(key)` is `O(capacity)`.
- Forgetting that `get` refreshes recency.
- Evicting before inserting when the key already exists, which throws out a key for no reason.
- Treating a stored value of `0` as a miss (`if not value`).
