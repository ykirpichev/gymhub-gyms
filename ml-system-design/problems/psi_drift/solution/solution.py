import bisect
import math


def _shares(values, edges):
    counts = [0] * (len(edges) + 1)
    for x in values:
        counts[bisect.bisect_right(edges, x)] += 1
    return [c / len(values) or 0.0001 for c in counts]


def solve(baseline, current, edges):
    e = _shares(baseline, edges)
    a = _shares(current, edges)
    psi = sum((ai - ei) * math.log(ai / ei) for ai, ei in zip(a, e))
    verdict = "stable" if psi < 0.1 else "moderate" if psi < 0.25 else "major"
    return psi, verdict
