#### The idea
Pinned messages (system prompt, rules) always go in, so subtract their cost from the budget first. If the rest fits,
send everything. Otherwise evict the oldest unpinned messages and replace them with one summary, trying the smallest
eviction first so as much recent context as possible survives verbatim.

<figure class="diagram"><svg viewBox="0 0 600 120" width="600" role="img" aria-label="Pinned messages, then a summary of evicted messages, then recent messages">
<rect x="10" y="20" width="90" height="36" rx="4" class="d-c"/><text x="55" y="43" class="d-text" text-anchor="middle">pinned</text>
<rect x="104" y="20" width="60" height="36" rx="4" class="d-c"/><text x="134" y="43" class="d-text" text-anchor="middle">pinned</text>
<rect x="190" y="20" width="200" height="36" rx="4" class="d-b"/><text x="290" y="43" class="d-text" text-anchor="middle">summary of the k oldest</text>
<rect x="400" y="20" width="90" height="36" rx="4" class="d-a"/><text x="445" y="43" class="d-text" text-anchor="middle">recent</text>
<rect x="494" y="20" width="90" height="36" rx="4" class="d-a"/><text x="539" y="43" class="d-text" text-anchor="middle">recent</text>
<text x="10" y="90" class="d-muted" text-anchor="start">Budget = pinned cost + summary cost + kept recent cost. Increase k until it fits.</text>
</svg>
<figcaption>What gets sent: pinned messages in order, one summary standing in for the evicted prefix, and the newest messages verbatim.</figcaption></figure>

Count each unpinned message once up front, so each candidate `k` costs one summary call plus a suffix sum.

#### A solution
```python
class SlidingWindowMemory:
    def __init__(self, count_tokens, summarize, budget):
        self.count_tokens = count_tokens
        self.summarize = summarize
        self.budget = budget

    def build(self, messages):
        pinned = [message for message in messages if message["pinned"]]
        rest = [message for message in messages if not message["pinned"]]
        pinned_cost = sum(self.count_tokens(message) for message in pinned)
        if pinned_cost > self.budget:
            raise ValueError("pinned messages exceed the token budget")
        room = self.budget - pinned_cost
        costs = [self.count_tokens(message) for message in rest]
        if sum(costs) <= room:
            return pinned + rest
        for k in range(1, len(rest) + 1):
            summary = self.summarize(rest[:k])
            if self.count_tokens(summary) + sum(costs[k:]) <= room:
                return pinned + [summary] + rest[k:]
        raise ValueError("no summary of the evicted messages fits in the budget")
```

#### What it costs
Up to `len(rest)` calls to `summarize`, each on a longer prefix. In production a summary is an LLM call, so you would
instead summarise incrementally (fold the newly evicted messages into the previous summary) or binary-search `k` if the
summary length grows with `k`. Prefix sums make each candidate's suffix cost `O(1)`.

#### Pitfalls
- Summarising pinned messages, or dropping them.
- Forgetting that the summary itself costs tokens.
- Evicting the newest messages instead of the oldest.
- Reordering output so pinned messages are not first.
- Mutating the input dicts, which the next `build` call reuses.
