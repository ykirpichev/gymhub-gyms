import math


def solve(x, chunk):
    if not x or chunk < 1:
        raise ValueError("need a non-empty x and chunk >= 1")
    m, l = -math.inf, 0.0
    for start in range(0, len(x), chunk):
        part = x[start:start + chunk]
        m_new = max(m, max(part))
        l = l * math.exp(m - m_new) + sum(math.exp(v - m_new) for v in part)
        m = m_new
    return (m, l, [math.exp(v - m) / l for v in x])
