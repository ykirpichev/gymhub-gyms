def solve(a, b, t):
    n, k, m = len(a), len(b), len(b[0])
    c = [[0] * m for _ in range(n)]
    loads = 0
    for i0 in range(0, n, t):
        for j0 in range(0, m, t):
            for k0 in range(0, k, t):
                loads += 2 * t * t
                for i in range(i0, i0 + t):
                    for j in range(j0, j0 + t):
                        acc = 0
                        for kk in range(k0, k0 + t):
                            acc += a[i][kk] * b[kk][j]
                        c[i][j] += acc
    return [c, loads]
