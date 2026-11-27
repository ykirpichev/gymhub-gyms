def _sample(d, v):
    total = 0.0
    last = None
    for j, prob in enumerate(d):
        if prob > 0:
            last = j
        total += prob
        if v < total:
            return j
    return last


def solve(draft, q, p, u, u_resample):
    out = []
    for i, x in enumerate(draft):
        if q[i][x] <= 0:
            raise ValueError("the draft cannot propose a token it gives probability 0")
        if u[i] < p[i][x] / q[i][x]:
            out.append(x)
            continue
        residual = [max(0.0, a - b) for a, b in zip(p[i], q[i])]
        norm = sum(residual)
        out.append(_sample([r / norm for r in residual], u_resample))
        return out
    out.append(_sample(p[len(draft)], u_resample))
    return out
