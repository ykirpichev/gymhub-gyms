def solve(relevance, similarity, k, lam):
    n = len(relevance)
    closest = [0.0] * n            # max similarity to anything already chosen
    chosen, taken = [], [False] * n
    for _ in range(min(k, n)):
        best, best_score = -1, None
        for i in range(n):
            if taken[i]:
                continue
            score = lam * relevance[i] - (1 - lam) * closest[i]
            if best_score is None or score > best_score:
                best, best_score = i, score
        chosen.append(best)
        taken[best] = True
        for i in range(n):
            closest[i] = max(closest[i], similarity[i][best])
    return chosen
