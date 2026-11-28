def solve(xs):
    v = list(xs)
    d, steps, adds = 1, 0, 0
    while d < len(v):
        v = [v[i] + v[i - d] if i >= d else v[i] for i in range(len(v))]
        adds += len(v) - d
        steps += 1
        d *= 2
    return [v, steps, adds]
