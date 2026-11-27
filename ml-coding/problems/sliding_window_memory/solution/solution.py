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
