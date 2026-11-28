import math

def solve(x):
    if x == 0:
        return 0.0
    a = abs(x)
    e = max(math.floor(math.log2(a)), -6)
    step = 2.0 ** (e - 3)
    q = min(round(a / step) * step, 448.0)
    return math.copysign(q, x)
