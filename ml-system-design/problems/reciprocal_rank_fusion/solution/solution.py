from collections import defaultdict


def solve(rankings, k, top_n):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1.0 / (k + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))[:top_n]
