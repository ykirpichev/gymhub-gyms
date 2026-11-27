import math


def _dcg(grades, k):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def solve(rankings, k):
    ndcgs, rrs = [], []
    for grades in rankings:
        ideal = _dcg(sorted(grades, reverse=True), k)
        ndcgs.append(_dcg(grades, k) / ideal if ideal > 0 else 0.0)
        rr = 0.0
        for i, g in enumerate(grades[:k]):
            if g > 0:
                rr = 1.0 / (i + 1)
                break
        rrs.append(rr)
    return {"ndcg": sum(ndcgs) / len(ndcgs), "mrr": sum(rrs) / len(rrs)}
